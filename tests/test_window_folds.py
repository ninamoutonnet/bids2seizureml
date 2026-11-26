"""Regression checks for continuous evaluation, patient eligibility, and folds.

Run with ``python -m unittest discover -s tests -v``. Only small synthetic
metadata tables are used; no EEG signals or optional test framework are needed.
"""

import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from bids2seizureml.folds import CVPlan, create_fold_dataframes
from bids2seizureml.segment import (
    SEGMENT_COLUMNS,
    build_evaluation_windows_dataframe,
    build_windowed_segments_dataframe,
)


def recording(centre="a", subject="01", run="01", duration=19.5, seizures=((5.0, 11.0),)):
    context = dict(
        dataset_root=f"/{centre}", centre=centre, subject=subject,
        session="01", run=run, recordingDuration=duration,
        events_path=f"/{centre}/sub-{subject}_run-{run}_events.tsv",
        signal_path=f"/{centre}/sub-{subject}_run-{run}_eeg.h5", signal_format="h5",
        edf_path=None, dateTime=None, channels=None,
    )
    rows = []
    cursor = 0.0
    for start, end in seizures:
        if start > cursor:
            rows.append(dict(context, start_s=cursor, end_s=start, label="background"))
        rows.append(dict(context, start_s=start, end_s=end, label="seizure"))
        cursor = end
    if cursor < duration:
        rows.append(dict(context, start_s=cursor, end_s=duration, label="background"))
    return pd.DataFrame(rows, columns=SEGMENT_COLUMNS)


def dataset():
    return pd.concat([
        recording(centre, str(subject), f"{run:02d}")
        for centre in ["a", "b"] for subject in range(3) for run in range(1, 6)
    ], ignore_index=True)


def patient_recordings(subject="01", seizure_runs=(2, 5, 8), n_recordings=10):
    return pd.concat([
        recording(subject=subject, run=f"{run:02d}",
                  seizures=((5.0, 11.0),) if run in seizure_runs else ())
        for run in range(1, n_recordings + 1)
    ], ignore_index=True)


def windows(segments):
    return (
        build_windowed_segments_dataframe(segments, 4.0, overlap_seizure=2.0),
        build_evaluation_windows_dataframe(segments, 4.0),
    )


def assigned_runs(result, split):
    """Read whole-recording assignments before background undersampling."""
    ids = result.cv_boundaries_df.loc[
        result.cv_boundaries_df["pool"] == split + "_pool", "recording_id"
    ]
    return set(result.recordings_df.loc[result.recordings_df["recording_id"].isin(ids), "run"])


class ContinuousWindowTests(unittest.TestCase):
    def test_boundary_windows_and_tail_keep_exact_seizure_coverage(self):
        segments = recording(duration=10.5, seizures=((2.0, 5.5), (8.5, 9.0)))
        result = build_evaluation_windows_dataframe(segments, 4.0)
        self.assertEqual(result["start_s"].tolist(), [0.0, 4.0, 8.0])
        self.assertEqual(result["end_s"].tolist(), [4.0, 8.0, 10.5])
        self.assertEqual(result["label"].tolist(), ["mixed"] * 3)
        np.testing.assert_allclose(result["seizure_duration_s"], [2.0, 1.5, 0.5])
        np.testing.assert_allclose(result["confidence_seizure"], [0.5, 0.375, 0.2])
        self.assertEqual(result["valid_duration_s"].sum(), 10.5)
        self.assertEqual(result["window_length_s"].tolist(), [4.0] * 3)

    def test_window_grid_does_not_depend_on_annotations(self):
        seizure = build_evaluation_windows_dataframe(recording(), 4.0, stride=2.0)
        background = build_evaluation_windows_dataframe(recording(seizures=()), 4.0, stride=2.0)
        assert_frame_equal(seizure[["start_s", "end_s"]], background[["start_s", "end_s"]])
        self.assertEqual(seizure.iloc[-1]["end_s"], 19.5)
        self.assertTrue((seizure["start_s"].to_numpy()[1:] <= seizure["end_s"].to_numpy()[:-1]).all())

    def test_short_recordings_and_missing_optional_identifiers_are_retained(self):
        segments = recording(duration=1.5, seizures=((0.0, 0.5),))
        segments[["session", "run"]] = None
        result = build_evaluation_windows_dataframe(segments, 4.0)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0]["valid_duration_s"], 1.5)
        self.assertEqual(result.iloc[0]["label"], "mixed")
        self.assertTrue(build_windowed_segments_dataframe(segments, 4.0).empty)

    def test_complete_seizure_and_background_labels(self):
        result = build_evaluation_windows_dataframe(
            recording(duration=12.0, seizures=((4.0, 8.0),)), 4.0
        )
        self.assertEqual(result["label"].tolist(), ["background", "seizure", "background"])
        self.assertEqual(result["confidence_seizure"].tolist(), [0.0, 1.0, 0.0])

    def test_invalid_lengths_and_gapped_strides_are_rejected(self):
        for length, stride in [(0, None), (float("nan"), None), (4, 0), (4, 5), (4, float("inf"))]:
            with self.subTest(length=length, stride=stride), self.assertRaises(ValueError):
                build_evaluation_windows_dataframe(recording(), length, stride=stride)

    def test_saved_training_and_evaluation_tables_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as directory:
            segments = recording()
            segments["events_path"] = str(Path(directory) / "example_events.tsv")
            build_windowed_segments_dataframe(segments, 4.0, save_per_recording=True)
            build_evaluation_windows_dataframe(segments, 4.0, save_per_recording=True)
            self.assertTrue((Path(directory) / "example_windows.csv").is_file())
            self.assertTrue((Path(directory) / "example_evaluation_windows.csv").is_file())


class FoldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.segments = dataset()
        cls.training, cls.evaluation = windows(cls.segments)

    def test_all_strategies_validation_modes_and_evaluation_modes(self):
        strategies = ["lopo", "kfold_patient", "leave_one_centre_out", "patient_specific"]
        for strategy, validation, evaluation_mode in itertools.product(
            strategies, [False, True], ["complete_recordings", "balanced_windows"]
        ):
            with self.subTest(strategy=strategy, validation=validation, mode=evaluation_mode):
                plan = CVPlan(strategy, validation=validation, eval_strategy=evaluation_mode, k=3)
                result = create_fold_dataframes(self.training, plan, self.evaluation)
                for fold_id, splits in result.folds.items():
                    self.assertEqual(splits["val"] is not None, validation)
                    active = [splits[name] for name in ["train", "val", "test"] if splits[name] is not None]
                    for left, right in itertools.combinations(active, 2):
                        self.assertTrue(set(left["recording_id"]).isdisjoint(right["recording_id"]))
                        if strategy != "patient_specific":
                            self.assertTrue(set(left["subject_key"]).isdisjoint(right["subject_key"]))
                    if strategy == "patient_specific":
                        self.assertLess(splits["train"]["recording_order"].max(), splits["test"]["recording_order"].min())
                        if validation:
                            self.assertLess(splits["train"]["recording_order"].max(), splits["val"]["recording_order"].min())
                            self.assertLess(splits["val"]["recording_order"].max(), splits["test"]["recording_order"].min())
                    if strategy == "leave_one_centre_out":
                        self.assertTrue(set(splits["train"]["centre"]).isdisjoint(splits["test"]["centre"]))
                    for name in ["val", "test"]:
                        if splits[name] is None or evaluation_mode != "complete_recordings":
                            continue
                        pool = name + "_pool"
                        ids = result.cv_boundaries_df.loc[
                            (result.cv_boundaries_df["fold_id"] == fold_id)
                            & (result.cv_boundaries_df["pool"] == pool), "recording_id"
                        ]
                        expected = result.evaluation_windows_df[
                            result.evaluation_windows_df["recording_id"].isin(ids)
                        ]
                        self.assertEqual(set(splits[name]["window_id"]), set(expected["window_id"]))
                    counts = splits["train"]["label"].value_counts()
                    self.assertEqual(counts["seizure"], counts["background"])
                if evaluation_mode == "complete_recordings":
                    self.assertGreater(result.fold_summary_df["n_mixed"].sum(), 0)
                    self.assertGreater(result.fold_summary_df["n_partial"].sum(), 0)

    def test_validation_fraction_counts_patients_not_recordings(self):
        result = create_fold_dataframes(
            self.training, CVPlan("lopo", validation=True, val_fraction=0.4), self.evaluation
        )
        for splits in result.folds.values():
            self.assertEqual(splits["val"]["subject_key"].nunique(), 2)
            self.assertEqual(splits["train"]["subject_key"].nunique(), 3)
            self.assertEqual(splits["val"]["recording_id"].nunique(), 10)

    def test_input_order_does_not_change_folds_or_sampling(self):
        plan = CVPlan("kfold_patient", validation=True, k=3)
        original = create_fold_dataframes(self.training, plan, self.evaluation)
        shuffled = create_fold_dataframes(
            self.training.sample(frac=1, random_state=4), plan,
            self.evaluation.sample(frac=1, random_state=9),
        )
        assert_frame_equal(original.fold_windows_df, shuffled.fold_windows_df)

    def test_short_recordings_without_pure_windows_are_kept_in_test(self):
        short = recording(run="06", duration=1.5, seizures=((0.0, 0.5),))
        segments = pd.concat([self.segments.query("centre == 'a' and subject == '0'"), short.assign(subject="0")])
        training, evaluation = windows(segments)
        result = create_fold_dataframes(training, CVPlan("patient_specific", validation=True), evaluation)
        test = next(iter(result.folds.values()))["test"]
        self.assertIn("06", set(test["run"]))
        short_rows = test[test["run"] == "06"]
        self.assertEqual(short_rows.iloc[0]["valid_duration_s"], 1.5)

    def test_held_out_annotations_cannot_change_training_selection(self):
        before = pd.concat([recording(run=f"{run:02d}") for run in range(1, 6)], ignore_index=True)
        after = pd.concat([
            before[before["run"] != "05"], recording(run="05", seizures=((4.0, 12.0),))
        ], ignore_index=True)
        plan = CVPlan("patient_specific", validation=True)
        selected = []
        for segments in [before, after]:
            training, evaluation = windows(segments)
            result = create_fold_dataframes(training, plan, evaluation)
            selected.append(next(iter(result.folds.values()))["train"])
        assert_frame_equal(selected[0], selected[1])

    def test_complete_recordings_requires_a_complete_evaluation_table(self):
        with self.assertRaisesRegex(ValueError, "requires evaluation_windows_df"):
            create_fold_dataframes(self.training, CVPlan("lopo"))
        missing_middle = self.evaluation.drop(self.evaluation.index[1])
        with self.assertRaisesRegex(ValueError, "complete regular grid"):
            create_fold_dataframes(self.training, CVPlan("lopo"), missing_middle)

    def test_insufficient_patients_produce_a_clear_error(self):
        training = self.training.query("centre == 'a' and subject in ['0', '1']")
        evaluation = self.evaluation.query("centre == 'a' and subject in ['0', '1']")
        with self.assertRaisesRegex(ValueError, "at least two non-test patients"):
            create_fold_dataframes(training, CVPlan("lopo", validation=True), evaluation)

    def test_insufficient_seizure_recordings_are_reported_before_sampling(self):
        segments = pd.concat([
            recording(run="01", seizures=()), recording(run="02", seizures=()), recording(run="03")
        ], ignore_index=True)
        training, evaluation = windows(segments)
        with self.assertLogs("bids2seizureml.folds", level="WARNING") as logs:
            with self.assertRaisesRegex(ValueError, "at least 3 usable seizure-containing"):
                create_fold_dataframes(training, CVPlan("patient_specific", validation=True), evaluation)
        self.assertIn("has 1 usable seizure-containing recording(s)", logs.output[0])

    def test_patient_specific_validation_includes_an_earlier_seizure(self):
        training, evaluation = windows(patient_recordings())
        for mode in ["balanced_windows", "complete_recordings"]:
            with self.subTest(mode=mode):
                plan = CVPlan("patient_specific", validation=True, eval_strategy=mode)
                result = create_fold_dataframes(training, plan, evaluation)
                splits = next(iter(result.folds.values()))
                expected = {"train": range(1, 5), "val": range(5, 8), "test": range(8, 11)}
                for split, runs in expected.items():
                    self.assertEqual(assigned_runs(result, split), {f"{run:02d}" for run in runs})
                    self.assertTrue(set(splits[split]["run"]).issubset(assigned_runs(result, split)))
                    self.assertTrue(splits[split]["label"].isin(["seizure", "mixed"]).any())
                shuffled = create_fold_dataframes(
                    training.sample(frac=1, random_state=3), plan,
                    evaluation.sample(frac=1, random_state=7),
                )
                assert_frame_equal(result.fold_windows_df, shuffled.fold_windows_df)

    def test_patient_specific_fraction_targets_leave_seizures_for_earlier_splits(self):
        training, evaluation = windows(patient_recordings(seizure_runs=(8, 9, 10)))
        for mode in ["balanced_windows", "complete_recordings"]:
            with self.subTest(mode=mode):
                result = create_fold_dataframes(training, CVPlan(
                    "patient_specific", validation=True, eval_strategy=mode,
                    test_fraction=0.8, val_fraction=0.8,
                ), evaluation)
                splits = next(iter(result.folds.values()))
                self.assertEqual(assigned_runs(result, "train"), {f"{run:02d}" for run in range(1, 9)})
                self.assertEqual(set(splits["val"]["run"]), {"09"})
                self.assertEqual(set(splits["test"]["run"]), {"10"})

    def test_patient_specific_without_validation_needs_only_two_seizure_recordings(self):
        training, evaluation = windows(patient_recordings(seizure_runs=(2, 8)))
        for mode in ["balanced_windows", "complete_recordings"]:
            with self.subTest(mode=mode):
                result = create_fold_dataframes(training, CVPlan(
                    "patient_specific", validation=False, eval_strategy=mode,
                ), evaluation)
                splits = next(iter(result.folds.values()))
                self.assertIsNone(splits["val"])
                self.assertEqual(assigned_runs(result, "train"), {f"{run:02d}" for run in range(1, 8)})
                self.assertEqual(assigned_runs(result, "test"), {"08", "09", "10"})

    def test_ineligible_patients_are_excluded_without_blocking_eligible_patients(self):
        segments = pd.concat([
            patient_recordings(subject="eligible"),
            patient_recordings(subject="two", seizure_runs=(2, 8)),
            patient_recordings(subject="none", seizure_runs=()),
            # Many events/windows in one recording still count as one recording.
            recording(subject="one", seizures=((1.0, 5.0), (7.0, 11.0), (13.0, 18.0))),
        ], ignore_index=True)
        training, evaluation = windows(segments)
        for mode in ["balanced_windows", "complete_recordings"]:
            with self.subTest(mode=mode):
                with self.assertLogs("bids2seizureml.folds", level="WARNING") as logs:
                    result = create_fold_dataframes(training, CVPlan(
                        "patient_specific", validation=True, eval_strategy=mode,
                    ), evaluation)
                self.assertEqual(len(result.folds), 1)
                self.assertEqual(set(result.fold_windows_df["subject"]), {"eligible"})
                self.assertEqual(len(logs.output), 3)
                self.assertTrue(all("Excluded" in line for line in logs.output))

    def test_short_seizures_are_usable_for_continuous_evaluation_only(self):
        segments = pd.concat([
            recording(run="01"),
            recording(run="02", duration=1.5, seizures=((0.0, 0.5),)),
            recording(run="03", duration=1.5, seizures=((0.0, 0.5),)),
        ], ignore_index=True)
        training, evaluation = windows(segments)
        result = create_fold_dataframes(training, CVPlan("patient_specific", validation=True), evaluation)
        splits = next(iter(result.folds.values()))
        self.assertEqual(set(splits["train"]["run"]), {"01"})
        self.assertEqual(splits["val"]["label"].tolist(), ["mixed"])
        self.assertEqual(splits["test"]["label"].tolist(), ["mixed"])
        for catalogue in [None, evaluation]:
            with self.subTest(continuous_catalogue=catalogue is not None):
                with self.assertLogs("bids2seizureml.folds", level="WARNING"):
                    with self.assertRaisesRegex(ValueError, "at least 3 usable seizure-containing"):
                        create_fold_dataframes(training, CVPlan(
                            "patient_specific", validation=True, eval_strategy="balanced_windows",
                        ), catalogue)

    def test_patient_specific_requires_usable_training_before_later_seizures(self):
        for long_training in [False, True]:
            with self.subTest(full_seizure_only_in_last_recording=long_training):
                segments = pd.concat([
                    recording(run="01", duration=1.5, seizures=((0.0, 0.5),)),
                    recording(run="02", duration=1.5, seizures=((0.0, 0.5),)),
                    recording(run="03") if long_training else
                    recording(run="03", duration=1.5, seizures=((0.0, 0.5),)),
                ], ignore_index=True)
                training, evaluation = windows(segments)
                with self.assertLogs("bids2seizureml.folds", level="WARNING") as logs:
                    with self.assertRaisesRegex(ValueError, "full seizure training windows"):
                        create_fold_dataframes(training, CVPlan("patient_specific", validation=True), evaluation)
                self.assertTrue(any("full seizure training windows" in line for line in logs.output))

    def test_other_cv_strategies_accept_patients_with_one_seizure_recording(self):
        segments = pd.concat([
            recording(centre=centre, subject=str(subject), run=f"{run:02d}",
                      seizures=((5.0, 11.0),) if run == 3 else ())
            for centre in ["a", "b"] for subject in range(3) for run in range(1, 4)
        ], ignore_index=True)
        training, evaluation = windows(segments)
        for strategy, mode in itertools.product(
            ["lopo", "kfold_patient", "leave_one_centre_out"],
            ["balanced_windows", "complete_recordings"],
        ):
            with self.subTest(strategy=strategy, mode=mode):
                result = create_fold_dataframes(training, CVPlan(
                    strategy, validation=True, k=3, eval_strategy=mode,
                ), evaluation)
                self.assertEqual(result.fold_windows_df["subject_key"].nunique(), 6)
                self.assertNotIn("has_training_seizure", result.recordings_df.columns)


if __name__ == "__main__":
    unittest.main()
