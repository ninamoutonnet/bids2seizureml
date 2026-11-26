"""Montage regression checks using tiny synthetic EEG recordings only."""

import unittest
from unittest.mock import patch

import numpy as np

from bids2seizureml.preprocessing import PreprocessingParams, PreprocessingPipeline
import mne


ELECTRODES = [
    "FP1", "F3", "C3", "P3", "O1", "F7", "T3", "T5", "FZ", "CZ", "PZ",
    "FP2", "F4", "C4", "P4", "O2", "F8", "T4", "T6",
]
LONGITUDINAL = [
    "FP1-F7", "F7-T3", "T3-T5", "T5-O1",
    "FP2-F8", "F8-T4", "T4-T6", "T6-O2",
    "FZ-CZ", "CZ-PZ", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]


def recording(names):
    data = np.random.default_rng(42).normal(size=(len(names), 32)) * 1e-6
    data -= data.mean(axis=0)
    return mne.io.RawArray(data, mne.create_info(names, 256, "eeg"), verbose="ERROR")


class MontageTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = PreprocessingPipeline.__new__(PreprocessingPipeline)
        self.pipeline.params = PreprocessingParams(dataset_path="/unused")

    def resolve(self, raw, target, reference=None):
        metadata = {} if reference is None else {"EEGReference": reference}
        return self.pipeline._resolve_montage(raw, target, metadata)

    def assert_skipped_unchanged(self, raw, target, reference=None):
        before = raw.get_data().copy()
        names = list(raw.ch_names)
        with self.assertLogs("bids2seizureml.preprocessing", level="WARNING"):
            result, label, skip = self.resolve(raw, target, reference)
        self.assertIs(result, raw)
        self.assertIsNone(label)
        self.assertTrue(skip)
        self.assertEqual(raw.ch_names, names)
        np.testing.assert_array_equal(raw.get_data(), before)

    def test_matching_longitudinal_aliases_pass_without_conversion(self):
        raw = recording(LONGITUDINAL)
        before = raw.get_data().copy()
        for reference in ["Longitudinal", "Double banana bipolar montage", "LONGITUDINAL (double banana)"]:
            for target in ["Longitudinal", "double-banana", "bipolar"]:
                with self.subTest(reference=reference, target=target):
                    with patch.object(self.pipeline, "_apply_montage") as convert:
                        result, label, skip = self.resolve(raw, target, reference)
                    convert.assert_not_called()
                    self.assertIs(result, raw)
                    self.assertEqual(label, "Longitudinal (double banana)")
                    self.assertFalse(skip)
        np.testing.assert_array_equal(raw.get_data(), before)

    def test_matching_average_aliases_pass_without_conversion(self):
        raw = recording(ELECTRODES)
        for reference in ["Common Average", "common_average", "COMMON-AVERAGE", "average", "AVG", "CAR"]:
            with self.subTest(reference=reference):
                with patch.object(self.pipeline, "_apply_montage") as convert:
                    result, label, skip = self.resolve(raw, "Common Average", reference)
                convert.assert_not_called()
                self.assertIs(result, raw)
                self.assertEqual(label, "Common Average")
                self.assertFalse(skip)

    def test_unknown_reference_is_not_assumed_average(self):
        for reference in [None, "n/a", "Unknown", "linked ears"]:
            with self.subTest(reference=reference):
                self.assert_skipped_unchanged(recording(ELECTRODES), "Common Average", reference)

    def test_no_target_keeps_unknown_reference(self):
        raw = recording(ELECTRODES)
        self.assertEqual(self.resolve(raw, None), (raw, None, False))

    def test_channel_suffix_inference(self):
        for names, expected in [
            (["FP1-AVG", "C3-AVG", "ECG"], "Common Average"),
            (["FP1-CZ", "C3-CZ"], "Cz-referential"),
            (LONGITUDINAL, "Longitudinal (double banana)"),
            (ELECTRODES, "Unknown"),
        ]:
            with self.subTest(names=names):
                self.assertEqual(self.pipeline._infer_montage(names, {}), expected)

    def test_longitudinal_conversion_has_all_pairs_and_correct_signals(self):
        synonyms = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
        for suffix, modern_names in [("", False), ("-AVG", False), ("-AVG", True)]:
            with self.subTest(suffix=suffix, modern_names=modern_names):
                names = [
                    (synonyms.get(name, name) if modern_names else name) + suffix
                    for name in ELECTRODES
                ]
                raw = recording(names)
                before = raw.get_data().copy()
                result, label, skip = self.resolve(raw, "Longitudinal", "Common Average")
                self.assertFalse(skip)
                self.assertIsNot(result, raw)
                self.assertEqual(label, "Longitudinal (double banana)")
                self.assertEqual(result.ch_names, LONGITUDINAL)
                expected = np.array([
                    before[ELECTRODES.index(anode)] - before[ELECTRODES.index(cathode)]
                    for anode, cathode in (name.split("-") for name in LONGITUDINAL)
                ])
                np.testing.assert_allclose(result.get_data(), expected, rtol=1e-12, atol=1e-20)
                self.assertEqual(raw.ch_names, names)
                np.testing.assert_array_equal(raw.get_data(), before)

    def test_cz_conversion_uses_available_scalp_channels(self):
        raw = recording(["FP1-AVG", "CZ-AVG", "C3-AVG", "ECG"])
        before = raw.get_data().copy()
        result, label, skip = self.resolve(raw, "Cz", "Common Average")
        self.assertFalse(skip)
        self.assertEqual(label, "Cz-referential")
        self.assertEqual(result.ch_names, ["FP1-CZ", "C3-CZ"])
        np.testing.assert_allclose(result.get_data(), before[[0, 2]] - before[1])
        np.testing.assert_array_equal(raw.get_data(), before)

    def test_missing_electrodes_are_detected_before_conversion(self):
        for target, missing in [("Longitudinal", "FP1"), ("Cz", "CZ")]:
            with self.subTest(target=target):
                raw = recording([name for name in ELECTRODES if name != missing])
                with patch("bids2seizureml.preprocessing.mne.set_bipolar_reference") as convert:
                    self.assert_skipped_unchanged(raw, target, "Common Average")
                convert.assert_not_called()

    def test_incomplete_existing_longitudinal_is_skipped(self):
        self.assert_skipped_unchanged(recording(LONGITUDINAL[:-1]), "Longitudinal", "Longitudinal")

    def test_duplicate_electrodes_are_skipped(self):
        raw = recording(ELECTRODES + ["FP1-AVG"])
        self.assert_skipped_unchanged(raw, "Longitudinal", "Common Average")

    def test_failed_conversion_keeps_original_signals(self):
        raw = recording(ELECTRODES)
        real_conversion = mne.set_bipolar_reference

        def fail_after_conversion(*args, **kwargs):
            converted = real_conversion(*args, **kwargs)
            converted._data[:] = 0
            raise RuntimeError("synthetic conversion failure")

        with patch("bids2seizureml.preprocessing.mne.set_bipolar_reference", side_effect=fail_after_conversion):
            self.assert_skipped_unchanged(raw, "Longitudinal", "Common Average")

    def test_unsupported_conversion_is_skipped(self):
        self.assert_skipped_unchanged(recording(["FP1-CZ", "C3-CZ"]), "Common Average", "Cz")

    def test_invalid_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported montage"):
            self.resolve(recording(ELECTRODES), "invalid", "Common Average")

    def test_metadata_uses_confirmed_reference_and_final_channels(self):
        raw = recording(ELECTRODES)
        for target in ["Common Average", "Longitudinal", "Cz"]:
            with self.subTest(target=target):
                result, label, skip = self.resolve(raw, target, "Common Average")
                self.assertFalse(skip)
                metadata = self.pipeline._build_output_metadata(result, {"EEGReference": "CAR"}, label)
                self.assertEqual(metadata["EEGReference"], label)
                self.assertEqual(metadata["Channels"], result.ch_names)
                self.assertEqual(metadata["EEGChannelCount"], len(result.ch_names))


if __name__ == "__main__":
    unittest.main()
