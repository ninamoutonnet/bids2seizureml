"""Small synthetic checks for EEG loading, shared stores and transform boundaries."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import torch
    from bids2seizureml import loaders
    from bids2seizureml.folds import CVPlan, create_fold_dataframes
    from bids2seizureml.segment import build_windowed_segments_dataframe, build_evaluation_windows_dataframe
    from bids2seizureml.transforms import normalise_eeg, gaussian_noise, build_augmentation_plan, apply_augmentation_plan


@unittest.skipUnless(HAS_TORCH, "Install PyTorch to run loader tests")
class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "recording.h5"
        self.sfreq = 256
        time = np.arange(int(10.5 * self.sfreq)) / self.sfreq
        self.data = np.vstack([20e-6 + 10e-6 * np.sin(2 * np.pi * 8 * time),
                               30e-6 + 20e-6 * np.cos(2 * np.pi * 12 * time)]).astype(np.float32)
        with h5py.File(self.path, "w") as h5:
            h5.attrs["sampling_frequency"] = self.sfreq
            h5.create_dataset("signals", data=self.data)
        self.rows = pd.DataFrame([
            dict(signal_path=str(self.path), signal_format="h5", start_s=start, end_s=end,
                 window_length_s=4.0, valid_duration_s=end-start, label=label,
                 confidence_seizure=fraction, subject="01", centre="synthetic")
            for start, end, label, fraction in [(0, 4, "background", 0),
                                               (4, 8, "seizure", 1), (8, 10.5, "mixed", 0.4)]
        ])

    def tearDown(self):
        loaders.close_cached_readers()
        self.temporary.cleanup()

    def folds(self):
        return {"a": {"train": self.rows.iloc[:2], "val": None, "test": self.rows},
                "b": {"train": self.rows.iloc[:2], "val": None, "test": self.rows}}

    def test_modes_share_unique_windows_and_match_lazy_data(self):
        baseline = loaders.build_datasets_from_folds(self.folds(), fold_name="a", normalisation=None)
        for mode in [None, "memory", "disk", "memmap"]:
            with self.subTest(mode=mode):
                result = loaders.build_dataloaders_from_folds(
                    self.folds(), batch_size=3, materialize=mode, cache_dir=self.root / "cache",
                    normalisation=None, return_metadata=True, shuffle_train=False,
                )
                x, y, meta = next(iter(result["a"]["test"]))
                self.assertEqual(tuple(x.shape), (3, 2, 1024))
                self.assertEqual(y.tolist(), [0, 1, -100])
                for index in range(3):
                    torch.testing.assert_close(x[index], baseline["test"][index][0])
                self.assertEqual(meta[-1]["valid_samples"], 640)
                self.assertEqual(meta[-1]["padding_samples"], 384)
                self.assertEqual(meta[-1]["confidence_seizure"], 0.4)
                self.assertTrue(torch.all(x[-1, :, 640:] == 0))
                if mode is not None:
                    left = result["a"]["train"].dataset.store
                    self.assertIs(left, result["b"]["test"].dataset.store)
                    self.assertEqual(len(left), 3)

    def test_cache_reuse_and_window_length_invalidation(self):
        for mode in ["disk", "memmap"]:
            options = dict(fold_name="a", materialize=mode, cache_dir=self.root / mode)
            first = loaders.build_datasets_from_folds(self.folds(), **options)
            with patch.object(loaders, "_read_window", side_effect=AssertionError("cache should be reused")):
                reused = loaders.build_datasets_from_folds(self.folds(), **options)
                torch.testing.assert_close(first["test"][2][0], reused["test"][2][0])
            with patch.object(loaders, "_read_window", wraps=loaders._read_window) as reader:
                loaders.build_datasets_from_folds(self.folds(), force_rebuild=True, **options)
                self.assertEqual(reader.call_count, 3)
            changed = self.rows.copy()
            changed["window_length_s"] = 5.0
            other = {"a": {"train": changed, "val": None, "test": changed}}
            rebuilt = loaders.build_datasets_from_folds(other, **options)
            self.assertEqual(rebuilt["test"][2][0].shape[1], 1280)

    def test_partial_normalisation_excludes_padding_in_every_mode(self):
        for mode in [None, "memory", "disk", "memmap"]:
            dataset = loaders.build_datasets_from_folds(
                self.folds(), fold_name="a", normalisation="zscore", materialize=mode,
                cache_dir=self.root / "normalised", return_metadata=True,
            )["test"]
            x, _, meta = dataset[2]
            valid = x[:, :meta["valid_samples"]].numpy()
            np.testing.assert_allclose(valid.mean(axis=1), 0, atol=1e-5)
            np.testing.assert_allclose(valid.std(axis=1), 1, atol=1e-5)
            self.assertTrue(torch.all(x[:, 640:] == 0))

    def test_default_global_zscore_uses_valid_samples_across_channels(self):
        with h5py.File(self.path, "r+") as h5:
            h5["signals"][0, :] = -1.0
            h5["signals"][1, :] = 3.0
        for mode in [None, "memory", "disk", "memmap"]:
            with self.subTest(mode=mode):
                options = dict(fold_name="a", materialize=mode,
                               cache_dir=self.root / "default_normalisation", return_metadata=True)
                datasets = loaders.build_datasets_from_folds(self.folds(), **options)
                batches = loaders.build_dataloaders_from_folds(self.folds(), **options)
                for dataset in [datasets["test"], batches["a"]["test"].dataset]:
                    x, _, meta = dataset[2]
                    valid = meta["valid_samples"]
                    # One shared mean/std preserves the difference between channels.
                    # Channel-wise z-score would turn both constant channels into zero.
                    np.testing.assert_allclose(x[0, :valid].numpy(), -1.0, atol=1e-6)
                    np.testing.assert_allclose(x[1, :valid].numpy(), 1.0, atol=1e-6)
                    self.assertEqual(valid, 640)
                    self.assertTrue(torch.all(x[:, valid:] == 0))

    def test_training_only_augmentation_does_not_modify_shared_store(self):
        augmentation = {"strategies": {"name": "amplitude_scaling", "min_scale": 2., "max_scale": 2.},
                        "copies": 1, "labels": ["seizure"], "seed": 7}
        for mode in [None, "memory", "disk", "memmap"]:
            datasets = loaders.build_datasets_from_folds(
                self.folds(), fold_name="a", normalisation=None, augmentation=augmentation,
                materialize=mode, cache_dir=self.root / "augmented", return_metadata=True,
            )
            self.assertEqual(len(datasets["train"]), 3)
            self.assertEqual(len(datasets["test"]), 3)
            original = datasets["train"][1][0].clone()
            augmented, _, meta = datasets["train"][2]
            torch.testing.assert_close(augmented, original * 2)
            torch.testing.assert_close(augmented, datasets["train"][2][0])
            torch.testing.assert_close(original, datasets["train"][1][0])
            self.assertTrue(meta["is_augmented"])
            self.assertFalse(datasets["test"][1][2]["is_augmented"])

    def test_edf_reads_volts_and_only_the_requested_interval(self):
        import mne
        raw = mne.io.RawArray(self.data, mne.create_info(["C3", "C4"], self.sfreq, "eeg"), verbose="ERROR")
        row = SimpleNamespace(**self.rows.iloc[-1].to_dict())
        row.signal_path, row.signal_format = str(self.root / "fixture.edf"), "edf"
        with patch.object(mne.io, "read_raw_edf", return_value=raw) as open_edf:
            with patch.object(raw, "get_data", wraps=raw.get_data) as read:
                x, count = loaders._read_window(row)
                read.assert_called_once_with(start=2048, stop=2688)
            self.assertFalse(open_edf.call_args.kwargs["preload"])
        np.testing.assert_allclose(x[:, :640], self.data[:, 2048:2688])
        self.assertEqual(count, 640)
        with self.assertRaises(ValueError):
            loaders._label_to_int("unexpected")

    def test_multiple_workers_and_seeded_shuffle(self):
        options = dict(fold_name="a", batch_size=2, materialize=None,
                       normalisation=None, num_workers=1, seed=17, return_metadata=True)
        first = loaders.build_dataloaders_from_folds(self.folds(), **options)["a"]["train"]
        second = loaders.build_dataloaders_from_folds(self.folds(), **options)["a"]["train"]
        try:
            left, right = next(iter(first)), next(iter(second))
            torch.testing.assert_close(left[0], right[0])
            torch.testing.assert_close(left[1], right[1])
        finally:
            for loader in (first, second):
                if loader._iterator is not None:
                    loader._iterator._shutdown_workers()

    @unittest.skipUnless(importlib.util.find_spec("pyedflib"), "Optional EDF writer needed for this fixture")
    def test_real_edf_matches_hdf5_voltage_units(self):
        from pyedflib import highlevel
        path = self.root / "voltage_units.edf"
        headers = [highlevel.make_signal_header(name, dimension="uV", sample_frequency=self.sfreq,
                                                physical_min=-100, physical_max=100)
                   for name in ["C3", "C4"]]
        highlevel.write_edf(str(path), self.data[:, :8*self.sfreq] * 1e6,
                            headers, highlevel.make_header(), digital=False)
        row = SimpleNamespace(**self.rows.iloc[1].to_dict())
        row.signal_path, row.signal_format = str(path), "edf"
        edf, valid = loaders._read_window(row)
        h5, expected_valid = loaders._read_window(SimpleNamespace(**self.rows.iloc[1].to_dict()))
        np.testing.assert_allclose(edf, h5, atol=4e-9)
        self.assertEqual(valid, expected_valid)

    def test_current_fold_tables_to_loader(self):
        segments = []
        for subject in ["01", "02", "03"]:
            for start, end, label in [(0, 4, "background"), (4, 8, "seizure"), (8, 10.5, "background")]:
                segments.append(dict(dataset_root=str(self.root), centre="synthetic", subject=subject,
                                     session="01", run="01", start_s=start, end_s=end, label=label,
                                     recordingDuration=10.5, channels=None, dateTime=None,
                                     events_path=str(self.root / f"{subject}_events.tsv"),
                                     signal_path=str(self.path), signal_format="h5", edf_path=None))
        frame = pd.DataFrame(segments)
        windows = build_windowed_segments_dataframe(frame, 4)
        for strategy in ["balanced_windows", "complete_recordings"]:
            evaluation = build_evaluation_windows_dataframe(frame, 4) if strategy == "complete_recordings" else None
            folds = create_fold_dataframes(windows, CVPlan("lopo", eval_strategy=strategy, validation=True), evaluation)
            result = loaders.build_dataloaders_from_folds(folds, batch_size=8, return_metadata=True)
            self.assertEqual(len(result), 3)
            x, y, metadata = next(iter(next(iter(result.values()))["test"]))
            self.assertEqual(tuple(x.shape[1:]), (2, 1024))
            self.assertEqual(len(metadata), len(y))

    def test_transform_options_and_voltage_scale_noise(self):
        for mode in [None, "zscore", "global_zscore", "minmax"]:
            self.assertEqual(normalise_eeg(self.data, mode).shape, self.data.shape)
        noise = gaussian_noise(self.data, np.random.RandomState(7), snr_db=20) - self.data
        snr = 10 * np.log10(np.mean(self.data ** 2) / np.mean(noise ** 2))
        self.assertAlmostEqual(float(snr), 20, delta=0.5)
        np.testing.assert_array_equal(gaussian_noise(np.zeros((2, 20)), np.random.RandomState(0)), 0)
        for name in ["gaussian_noise", "amplitude_scaling", "time_shift", "channel_dropout"]:
            plan = build_augmentation_plan([SimpleNamespace(label="seizure")], {"strategies": name})
            one, two = (apply_augmentation_plan(self.data, plan[1]) for _ in range(2))
            np.testing.assert_array_equal(one, two)


if __name__ == "__main__":
    unittest.main()
