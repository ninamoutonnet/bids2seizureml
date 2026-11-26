"""Small synthetic checks for filter failures, validation, and resampling."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np

from bids2seizureml.preprocessing import PreprocessingParams, PreprocessingPipeline
import mne


MODULE = "bids2seizureml.preprocessing"


def recording(sfreq=256, n_samples=2048):
    data = np.random.default_rng(42).normal(size=(2, n_samples)) * 1e-6
    return mne.io.RawArray(data, mne.create_info(["C3", "C4"], sfreq, "eeg"), verbose="ERROR")


class SignalProcessingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()

    def pipeline(self, directory=None, **options):
        return PreprocessingPipeline(PreprocessingParams(
            dataset_path=directory or self.root, output_format="h5", **options
        ))

    def test_sampling_and_notch_frequencies_must_be_finite_and_positive(self):
        for name in ["target_sampling_freq", "notch_filter"]:
            for value in [0, -1, float("nan"), float("inf")]:
                with self.subTest(name=name, value=value):
                    pipeline = self.pipeline(**{name: value})
                    with self.assertRaisesRegex(ValueError, name):
                        pipeline._validate_parameters()

    def test_bandpass_cutoffs_must_be_finite_positive_and_ordered(self):
        for cutoffs in [(0, 40), (-1, 40), (40, 40), (50, 40), (1, float("inf")), (float("nan"), 40)]:
            with self.subTest(cutoffs=cutoffs):
                with self.assertRaisesRegex(ValueError, "bandpass_filter"):
                    self.pipeline(bandpass_filter=cutoffs)._validate_parameters()

    def test_filter_limits_use_the_rate_after_resampling(self):
        for options in [dict(notch_filter=64), dict(notch_filter=80), dict(bandpass_filter=(1, 64))]:
            with self.subTest(options=options):
                pipeline = self.pipeline(target_sampling_freq=128, **options)
                with patch.object(pipeline, "_find_all_edf_files") as discover:
                    with self.assertRaisesRegex(ValueError, "below 64 Hz"):
                        pipeline.run()
                discover.assert_not_called()
        self.pipeline(target_sampling_freq=256, notch_filter=60, bandpass_filter=(1, 100))._validate_parameters()

    def test_without_resampling_filter_limits_use_each_recording_rate(self):
        for options in [dict(notch_filter=60), dict(bandpass_filter=(1, 60))]:
            with self.subTest(options=options):
                pipeline = self.pipeline(**options)
                pipeline._validate_parameters()  # The input rate is not known yet.
                raw = recording(sfreq=100)
                path = self.root / "low_rate_eeg.edf"
                with patch(f"{MODULE}.mne.io.read_raw_edf", return_value=raw):
                    with self.assertLogs(MODULE, level="WARNING") as logs:
                        out = pipeline._process_single_file(path)
                self.assertFalse(out.exists())
                self.assertFalse(pipeline.output_dir.exists())
                self.assertIn("frequency validation failed", logs.output[0])
                self.assertIn(str(path), logs.output[0])

    def test_failed_operation_does_not_write_signals_or_metadata(self):
        operations = [("resample", "resampling"), ("notch_filter", "notch filter"), ("filter", "bandpass filter")]
        for method, label in operations:
            with self.subTest(method=method):
                pipeline = self.pipeline(target_sampling_freq=128, notch_filter=50, bandpass_filter=(1, 40))
                raw = recording()
                path = self.root / "failed_eeg.edf"

                def fail(*args, **kwargs):
                    raw._data[:] = np.nan  # Even a partial in-memory change must never be exported.
                    raise RuntimeError("synthetic failure")

                with ExitStack() as stack:
                    stack.enter_context(patch(f"{MODULE}.mne.io.read_raw_edf", return_value=raw))
                    for name, _ in operations:
                        stack.enter_context(patch.object(raw, name, side_effect=fail if name == method else None, return_value=raw))
                    writer = stack.enter_context(patch.object(pipeline, "_write_processed_recording"))
                    with self.assertLogs(MODULE, level="WARNING") as logs:
                        out = pipeline._process_single_file(path)
                writer.assert_not_called()
                self.assertFalse(out.exists())
                self.assertFalse(out.with_suffix(".json").exists())
                self.assertIn(str(path), logs.output[0])
                self.assertIn(f"{label} failed: synthetic failure", logs.output[0])

    def test_serial_and_parallel_runs_continue_after_a_failed_recording(self):
        for parallel in [False, True]:
            with self.subTest(parallel=parallel):
                directory = self.root / str(parallel)
                directory.mkdir()
                bad_path, good_path = directory / "bad_eeg.edf", directory / "good_eeg.edf"
                bad_path.touch()
                good_path.touch()
                good_path.with_suffix(".json").write_text(json.dumps({
                    "PowerLineFrequency": 50,
                    "SoftwareFilters": {"high-pass filter": {"half-amplitude cutoff (Hz)": 0.5}},
                }))
                bad_raw, good_raw = recording(), recording()
                bad_raw.notch_filter = Mock(side_effect=RuntimeError("broken filter"))
                recordings = {str(bad_path): bad_raw, str(good_path): good_raw}
                pipeline = self.pipeline(directory, notch_filter=60, parallel=parallel, n_jobs=2)

                with ExitStack() as stack:
                    stack.enter_context(patch(f"{MODULE}.mne.io.read_raw_edf", side_effect=lambda path, **kw: recordings[path]))
                    stack.enter_context(patch(f"{MODULE}.tqdm", None))
                    # Exercise the worker and futures path while keeping the synthetic reader in this process.
                    stack.enter_context(patch(f"{MODULE}.ProcessPoolExecutor", ThreadPoolExecutor))
                    with self.assertLogs(MODULE, level="WARNING") as logs:
                        pipeline.run()

                self.assertFalse(pipeline._construct_output_path(bad_path).exists())
                good_output = pipeline._construct_output_path(good_path)
                self.assertTrue(pipeline._is_valid_output(good_output))
                metadata = json.loads(good_output.with_suffix(".json").read_text())
                self.assertIn("notch filter", metadata["SoftwareFilters"])
                self.assertEqual(metadata["PowerLineFrequency"], 50)
                self.assertEqual(metadata["SoftwareFilters"]["high-pass filter"]["half-amplitude cutoff (Hz)"], 0.5)
                with h5py.File(good_output, "r") as h5:
                    self.assertEqual(json.loads(h5["metadata_json"][()]), metadata)
                report = json.loads((pipeline.output_dir / "PREPROCESSING_LOG.json").read_text())
                self.assertEqual(report["num_created_this_run"], 1)
                self.assertEqual(report["num_output_files"], 1)
                self.assertTrue(any(str(bad_path) in message and "broken filter" in message for message in logs.output))

    def test_repeated_filter_types_preserve_all_previous_settings(self):
        inherited = {
            "PowerLineFrequency": 50,
            "SoftwareFilters": {
                "high-pass filter": {"half-amplitude cutoff (Hz)": 0.5},
                "high-pass filter (preprocessing 1)": {"half-amplitude cutoff (Hz)": 0.75},
                "low-pass filter": {"half-amplitude cutoff (Hz)": 100},
                "notch filter": {"half-amplitude cutoff (Hz)": 50},
            },
        }
        original = deepcopy(inherited)
        pipeline = self.pipeline(bandpass_filter=(1, 40), notch_filter=60)
        metadata = pipeline._build_output_metadata(recording(), inherited, None)
        filters = metadata["SoftwareFilters"]
        self.assertEqual(metadata["PowerLineFrequency"], 50)
        self.assertEqual(inherited, original)
        self.assertEqual(len(filters), len(inherited["SoftwareFilters"]) + 3)
        for name, details in inherited["SoftwareFilters"].items():
            self.assertEqual(filters[name], details)
        for name, cutoff in [("high-pass filter", 1), ("low-pass filter", 40), ("notch filter", 60)]:
            self.assertTrue(any(
                key.startswith(name) and details["half-amplitude cutoff (Hz)"] == cutoff
                for key, details in filters.items()
            ))

    def test_missing_power_line_frequency_is_not_guessed_from_notch(self):
        pipeline = self.pipeline(notch_filter=60)
        for inherited in [{}, {"PowerLineFrequency": "n/a"}]:
            with self.subTest(inherited=inherited):
                metadata = pipeline._build_output_metadata(recording(), inherited, None)
                self.assertEqual(metadata["PowerLineFrequency"], "n/a")

    def test_text_filter_history_is_retained_when_adding_a_filter(self):
        history = "Acquisition software applied a 0.5 Hz high-pass filter."
        metadata = self.pipeline(notch_filter=60)._build_output_metadata(
            recording(), {"SoftwareFilters": history}, None
        )
        self.assertIn(history, metadata["SoftwareFilters"].values())
        self.assertIn("notch filter", metadata["SoftwareFilters"])

    def test_no_new_filters_preserves_inherited_filter_metadata(self):
        for inherited in [{}, {"SoftwareFilters": "n/a"}, {"SoftwareFilters": {"notch filter": {"frequency": 50}}}]:
            with self.subTest(inherited=inherited):
                metadata = self.pipeline()._build_output_metadata(recording(), inherited, None)
                self.assertEqual(metadata["SoftwareFilters"], inherited.get("SoftwareFilters", "n/a"))

    def test_successful_resampling_preserves_duration_and_event_times(self):
        raw = recording(sfreq=250, n_samples=1003)
        raw.set_annotations(mne.Annotations(onset=[0.5], duration=[0.75], description=["seizure"]))
        duration = raw.n_times / raw.info["sfreq"]
        path = self.root / "recording_eeg.edf"
        events = self.root / "recording_events.tsv"
        event_text = "onset\tduration\teventType\n0.5\t0.75\tsz\n"
        events.write_text(event_text)
        pipeline = self.pipeline(target_sampling_freq=128)
        with patch(f"{MODULE}.mne.io.read_raw_edf", return_value=raw):
            out = pipeline._process_single_file(path)

        with h5py.File(out, "r") as h5:
            self.assertEqual(h5.attrs["sampling_frequency"], 128)
            self.assertLessEqual(abs(h5["signals"].shape[1] - duration * 128), 1)
            self.assertLessEqual(abs(h5.attrs["duration_s"] - duration), 1 / 128)
            metadata = json.loads(h5["metadata_json"][()])
            self.assertEqual(metadata["SamplingFrequency"], 128)
            self.assertEqual(metadata["RecordingDuration"], h5.attrs["duration_s"])
            self.assertEqual(h5["events_tsv"][()].decode(), event_text)
        np.testing.assert_allclose(raw.annotations.onset, [0.5])
        np.testing.assert_allclose(raw.annotations.duration, [0.75])
        self.assertEqual((out.parent / events.name).read_text(), event_text)

    def test_resume_restores_missing_json_and_events_without_processing_eeg(self):
        for parallel in [False, True]:
            with self.subTest(parallel=parallel):
                directory = self.root / str(parallel)
                directory.mkdir()
                source = directory / "recording_eeg.edf"
                source.touch()
                events = source.with_name("recording_events.tsv")
                events.write_text("onset\tduration\teventType\n0.5\t0.75\tsz\n")
                pipeline = self.pipeline(directory, parallel=parallel, n_jobs=2)
                raw = recording(n_samples=64)
                metadata = pipeline._build_output_metadata(raw, {"PowerLineFrequency": 50}, None)
                out = pipeline._construct_output_path(source)
                pipeline._write_processed_recording(raw, out, metadata, source, events)
                # Simulate interruption after the EEG was saved, before its companions.
                out.with_suffix(".json").unlink()
                dest_events = out.parent / events.name
                dest_events.with_name(f"{dest_events.name}.tmp").write_text("interrupted copy")
                signal_bytes = out.read_bytes()
                signal_mtime = out.stat().st_mtime_ns

                with ExitStack() as stack:
                    process = stack.enter_context(patch.object(pipeline, "_process_single_file", side_effect=AssertionError("EEG must be reused")))
                    reader = stack.enter_context(patch(f"{MODULE}.mne.io.read_raw_edf", side_effect=AssertionError("EEG must not be loaded")))
                    stack.enter_context(patch(f"{MODULE}.tqdm", None))
                    executor = stack.enter_context(patch(f"{MODULE}.ProcessPoolExecutor", wraps=ThreadPoolExecutor))
                    pipeline.run()
                process.assert_not_called()
                reader.assert_not_called()
                if parallel:
                    self.assertEqual(executor.call_count, 1)
                self.assertEqual(out.read_bytes(), signal_bytes)
                self.assertEqual(out.stat().st_mtime_ns, signal_mtime)
                self.assertEqual(json.loads(out.with_suffix(".json").read_text()), metadata)
                self.assertEqual(dest_events.read_bytes(), events.read_bytes())
                self.assertFalse(dest_events.with_name(f"{dest_events.name}.tmp").exists())
                report = json.loads((pipeline.output_dir / "PREPROCESSING_LOG.json").read_text())
                self.assertEqual(report["num_created_this_run"], 0)
                self.assertEqual(report["num_skipped_existing"], 1)

    def test_event_recovery_uses_embedded_hdf5_events_if_source_is_missing(self):
        for has_events in [True, False]:
            with self.subTest(has_events=has_events):
                directory = self.root / str(has_events)
                directory.mkdir()
                source = directory / "recording_eeg.edf"
                events = source.with_name("recording_events.tsv")
                text = "onset\tduration\teventType\n0.5\t0.75\tsz\n"
                if has_events:
                    events.write_text(text)
                pipeline = self.pipeline(directory)
                raw = recording(n_samples=64)
                out = pipeline._construct_output_path(source)
                pipeline._write_processed_recording(raw, out, {}, source, events)
                if has_events:
                    events.unlink()
                pipeline._ensure_companion_files(source, out)
                restored = out.parent / events.name
                if has_events:
                    self.assertEqual(restored.read_text(), text)
                else:
                    self.assertFalse(restored.exists())

    def test_existing_companion_files_are_preserved_for_both_output_formats(self):
        for output_format in ["edf", "h5"]:
            with self.subTest(output_format=output_format):
                source = self.root / "recording_eeg.edf"
                source.with_name("recording_events.tsv").write_text("source annotations")
                out_dir = self.root / output_format
                out_dir.mkdir()
                out = out_dir / f"processed.{output_format}"
                out.with_suffix(".json").write_text('{"existing": true}')
                dest_events = out_dir / "recording_events.tsv"
                dest_events.write_text("existing annotations")
                pipeline = self.pipeline()
                pipeline._ensure_companion_files(source, out)
                self.assertEqual(out.with_suffix(".json").read_text(), '{"existing": true}')
                self.assertEqual(dest_events.read_text(), "existing annotations")


if __name__ == "__main__":
    unittest.main()
