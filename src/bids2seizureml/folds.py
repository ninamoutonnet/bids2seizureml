"""Dataframe-first fold construction for EEG seizure experiments.

This module owns only the metadata side of cross-validation. It turns a window
metadata dataframe into deterministic, inspectable tables that describe which
windows belong to train, validation, and test splits.

It deliberately does not load EEG arrays, normalise signals, or apply
augmentation. Those steps should consume the returned fold dataframes.
"""

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# This module consumes the window table produced by
# ``segment.build_windowed_segments_dataframe`` (see that module's
# ``WINDOW_COLUMNS``). REQUIRED_WINDOW_COLUMNS lists only the columns folds
# strictly needs; ``centre`` and the two ``nearest_*_distance_s`` columns are
# optional and guarded where used.
CV_STRATEGIES = frozenset(
    {
        "lopo",
        "kfold_patient",
        "leave_one_centre_out",
        "patient_specific",
    }
)
EVAL_STRATEGIES = frozenset({"balanced_windows", "complete_recordings"})
# Legacy string values remain accepted by _coerce_plan.
VALIDATION_POLICIES = frozenset({"none", "use_validation"})

REQUIRED_WINDOW_COLUMNS = [
    "dataset_root",
    "subject",
    "session",
    "run",
    "start_s",
    "end_s",
    "label",
]

SORT_WINDOW_COLUMNS = [
    "centre",
    "subject",
    "recording_order",
    "session",
    "run",
    "start_s",
    "end_s",
    "label",
]

CATALOG_RECORDING_COLUMNS = [
    "dataset_root",
    "centre",
    "subject",
    "session",
    "run",
]

__all__ = [
    "CV_STRATEGIES",
    "EVAL_STRATEGIES",
    "VALIDATION_POLICIES",
    "CVPlan",
    "FoldDataFrames",
    "build_catalog_tables",
    "assign_cv_boundaries",
    "make_balanced_reference",
    "resolve_fold_windows",
    "fold_windows_to_split_dict",
    "create_fold_dataframes",
]


@dataclass(frozen=True)
class CVPlan:
    """Public configuration for dataframe fold creation.

    Parameters
    ----------
    strategy:
        One of {"lopo", "kfold_patient", "leave_one_centre_out",
        "patient_specific"}. Patient-specific experiments use one chronological
        split per patient and require three usable seizure-containing recordings
        with validation, or two without it. Each enabled split contains seizure
        data; training requires full, pure seizure windows. Ineligible patients
        are excluded with a warning.
    eval_strategy:
        "complete_recordings" uses continuous windows for validation and test
        and requires evaluation_windows_df. "balanced_windows" samples pure
        windows separately within each split.
    validation:
        Whether to hold out a validation set (True/False). When True, a
        validation set is carved from the training side (for patient_specific,
        a chronological middle block; for the other strategies, whole
        patients held out from the non-test patients).
    val_fraction:
        Fraction of non-test patients to hold out when validation=True.
        For patient_specific, the target fraction of pre-test recordings assigned
        to the middle block. Its boundary is adjusted to retain seizure data in
        both training and validation.
    k:
        Number of folds for strategy="kfold_patient".
    seed:
        Seed for deterministic subject grouping, background selection, and
        validation holdout.
    background_to_seizure_ratio:
        Number of background windows selected per seizure window for the
        balanced reference table.
    min_background_seizure_distance_s:
        Minimum distance in seconds required between a background window and
        the nearest seizure start/end when creating the balanced reference
        table. Set to 0.0 to allow all background windows.
    min_background_recording_edge_distance_s:
        Minimum distance in seconds required between a background window and
        the nearest recording start/end when creating the balanced reference
        table. Set to 0.0 to allow recording-edge background windows.
    test_fraction:
        Target fraction of latest recordings held out for patient_specific.
        Its boundary is adjusted to include a seizure in test while leaving
        seizure data for earlier splits. Fractions count recordings, not hours
        or seizures. Other strategies determine their test set from the CV
        boundary.
    """

    strategy: str
    eval_strategy: str = "complete_recordings"
    validation: bool = False
    val_fraction: float = 0.20
    k: int = 5
    seed: int = 42
    background_to_seizure_ratio: float = 1.0
    min_background_seizure_distance_s: float = 0.0
    min_background_recording_edge_distance_s: float = 0.0
    test_fraction: float = 0.20


@dataclass
class FoldDataFrames:
    """Container returned by ``create_fold_dataframes``."""

    recordings_df: pd.DataFrame
    windows_df: pd.DataFrame
    cv_boundaries_df: pd.DataFrame
    balanced_reference_df: pd.DataFrame
    fold_windows_df: pd.DataFrame
    fold_summary_df: pd.DataFrame
    folds: Dict[str, Dict[str, Optional[pd.DataFrame]]]
    evaluation_windows_df: Optional[pd.DataFrame] = None


def create_fold_dataframes(
    windows_df: pd.DataFrame,
    plan: Union[CVPlan, dict],
    evaluation_windows_df: Optional[pd.DataFrame] = None,
) -> FoldDataFrames:
    """Run the complete dataframe-only fold pipeline.

    Returns every intermediate table plus a convenience ``folds`` dictionary:

    ``folds[fold_id]["train"]``, ``folds[fold_id]["val"]``, ``folds[fold_id]["test"]``.
    ``windows_df`` contains pure training candidates. For complete-recording
    evaluation, supply the output of ``build_evaluation_windows_dataframe``;
    its full recording catalogue defines the split boundaries before sampling.
    """

    plan = _coerce_plan(plan)
    _validate_plan(plan)
    recordings_df, prepared_windows_df = build_catalog_tables(windows_df)
    prepared_evaluation_df = None
    if plan.eval_strategy == "complete_recordings" and evaluation_windows_df is None:
        raise ValueError(
            "complete_recordings requires evaluation_windows_df from "
            "build_evaluation_windows_dataframe"
        )
    if evaluation_windows_df is not None:
        recordings_df, prepared_evaluation_df = build_catalog_tables(evaluation_windows_df)
        _validate_evaluation_coverage(prepared_evaluation_df)
        if not set(prepared_windows_df["recording_id"]).issubset(recordings_df["recording_id"]):
            raise ValueError("evaluation_windows_df is missing recordings present in windows_df")
        # Both tables must use the full catalogue's order, including recordings
        # with no pure training windows. Acquisition-order assumptions are unchanged.
        prepared_windows_df = prepared_windows_df.drop(columns="recording_order").merge(
            recordings_df[["recording_id", "recording_order"]], on="recording_id", how="left"
        )
    if plan.strategy == "patient_specific":
        # A short seizure may appear only in mixed evaluation windows. It is
        # usable for continuous evaluation, but cannot supply pure training windows.
        training_seizure_ids = prepared_windows_df.loc[
            prepared_windows_df["label"] == "seizure", "recording_id"
        ]
        recordings_df["has_training_seizure"] = recordings_df["recording_id"].isin(
            training_seizure_ids
        )
    cv_boundaries_df = assign_cv_boundaries(recordings_df, plan)
    balanced_reference_df = make_balanced_reference(prepared_windows_df, plan, cv_boundaries_df)
    fold_windows_df, fold_summary_df = resolve_fold_windows(
        cv_boundaries_df=cv_boundaries_df,
        windows_df=prepared_windows_df,
        balanced_reference_df=balanced_reference_df,
        plan=plan,
        evaluation_windows_df=prepared_evaluation_df,
    )
    folds = fold_windows_to_split_dict(fold_windows_df)
    return FoldDataFrames(
        recordings_df=recordings_df,
        windows_df=prepared_windows_df,
        cv_boundaries_df=cv_boundaries_df,
        balanced_reference_df=balanced_reference_df,
        fold_windows_df=fold_windows_df,
        fold_summary_df=fold_summary_df,
        folds=folds,
        evaluation_windows_df=prepared_evaluation_df,
    )


def build_catalog_tables(windows_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create stable recording/window IDs and one-row-per-recording catalog.

    The returned window dataframe is a copy with added ``subject_key``,
    ``recording_id``, ``recording_order``, and ``window_id`` columns.
    """

    if windows_df is None or windows_df.empty:
        logger.warning("windows_df is empty; fold tables will be empty")
        empty_windows = pd.DataFrame(
            columns=list(REQUIRED_WINDOW_COLUMNS)
            + ["centre", "subject_key", "recording_id", "recording_order", "window_id"]
        )
        empty_recordings = pd.DataFrame(
            columns=["recording_id", "subject_key", "recording_order"] + CATALOG_RECORDING_COLUMNS
        )
        return empty_recordings, empty_windows

    _require_columns(windows_df, REQUIRED_WINDOW_COLUMNS)
    df = windows_df.copy().reset_index(drop=True)

    if "centre" not in df.columns:
        logger.info("windows_df has no 'centre' column; defaulting centre to 'unknown'")
        df["centre"] = "unknown"
    if "recording_id" in df.columns:
        df = df.drop(columns=["recording_id"])

    df["subject_key"] = _subject_key_series(df)

    rec_key_cols = [c for c in CATALOG_RECORDING_COLUMNS if c in df.columns]
    recording_keys = df[rec_key_cols].drop_duplicates().reset_index(drop=True)
    recording_keys["recording_id"] = _recording_ids(recording_keys)
    df = df.merge(recording_keys[rec_key_cols + ["recording_id"]], on=rec_key_cols, how="left")

    if "window_id" not in df.columns:
        window_key_cols = rec_key_cols + ["start_s", "end_s", "label"]
        df["window_id"] = "win_" + _hash_rows(df, window_key_cols).str.slice(0, 12)

    optional_cols = [
        "edf_path",
        "events_path",
        "dateTime",
        "recordingDuration",
    ]
    recording_cols = (
        ["recording_id", "subject_key"]
        + rec_key_cols
        + [c for c in optional_cols if c in df.columns]
    )
    recordings_df = (
        df[recording_cols]
        .drop_duplicates(subset=["recording_id"])
        .reset_index(drop=True)
    )
    # Seizure presence includes mixed evaluation windows. Patient-specific
    # boundaries use it to retain seizures in their held-out blocks.
    seizure_by_recording = (
        df.assign(_is_seizure=df["label"].isin(["seizure", "mixed"]))
        .groupby("recording_id")["_is_seizure"]
        .any()
        .rename("has_seizure")
    )
    recordings_df = recordings_df.merge(
        seizure_by_recording, on="recording_id", how="left"
    )
    recordings_df["has_seizure"] = recordings_df["has_seizure"].fillna(False).astype(bool)
    recordings_df = _add_recording_order(recordings_df)
    recordings_df = _sort_recordings(recordings_df).reset_index(drop=True)
    recording_order = recordings_df[["recording_id", "recording_order"]]
    df = df.drop(columns=["recording_order"], errors="ignore")
    df = df.merge(recording_order, on="recording_id", how="left")
    df = _sort_windows(df).reset_index(drop=True)
    return recordings_df, df


def assign_cv_boundaries(recordings_df: pd.DataFrame, plan: Union[CVPlan, dict]) -> pd.DataFrame:
    """Assign recordings to train/test pools according to the CV strategy.

    Validation is assigned before window sampling: a chronological middle
    recording block for patient_specific, and separate patients otherwise.
    Patient-specific catalogs must include has_seizure. The optional
    has_training_seizure flag identifies recordings with pure seizure windows;
    otherwise has_seizure is assumed to describe usable training data too.
    create_fold_dataframes supplies both flags automatically.
    """

    plan = _coerce_plan(plan)
    _validate_plan(plan)
    if recordings_df is None or recordings_df.empty:
        return _empty_boundaries_df()

    _require_columns(recordings_df, ["recording_id", "subject_key", "centre"])
    recs = _sort_recordings(recordings_df).reset_index(drop=True)

    if plan.strategy == "patient_specific":
        return _assign_patient_specific(recs, plan)
    assigners = {
        "lopo": _assign_lopo,
        "kfold_patient": _assign_kfold_patient,
        "leave_one_centre_out": _assign_leave_one_centre_out,
    }
    boundaries = assigners[plan.strategy](recs, plan)
    return _assign_validation_patients(boundaries, plan)


def make_balanced_reference(
    windows_df: pd.DataFrame,
    plan: Union[CVPlan, dict],
    cv_boundaries_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Create fixed selected windows per subject.

    All seizure windows are retained. Background windows are sampled
    deterministically within each ``subject_key`` according to
    ``background_to_seizure_ratio``. When
    ``min_background_seizure_distance_s`` or
    ``min_background_recording_edge_distance_s`` is positive, background
    candidates too close to seizure boundaries or recording edges are excluded
    before sampling. Subjects with no seizure windows are not included in the
    balanced reference. When boundaries are supplied, sampling is performed
    separately within each fold/pool, so held-out seizure counts never control
    training sample selection. Continuous evaluation pools are not sampled.
    """

    plan = _coerce_plan(plan)
    _validate_plan(plan)
    columns = [
        "selection_id",
        "subject_key",
        "window_id",
        "label",
        "role",
        "selected_order",
    ]
    if cv_boundaries_df is not None:
        parts = []
        for (fold_id, pool), group in cv_boundaries_df.groupby(["fold_id", "pool"], sort=True):
            if pool != "train_pool" and plan.eval_strategy == "complete_recordings":
                continue
            candidates = _windows_for_recordings(windows_df, group["recording_id"])
            reference = make_balanced_reference(candidates, plan)
            reference.insert(0, "pool", pool)
            reference.insert(0, "fold_id", fold_id)
            parts.append(reference)
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["fold_id", "pool"] + columns
        )
    if windows_df is None or windows_df.empty:
        return pd.DataFrame(columns=columns)

    _require_columns(windows_df, ["subject_key", "window_id", "label"])
    if plan.background_to_seizure_ratio < 0:
        raise ValueError("background_to_seizure_ratio must be >= 0")

    df = windows_df[windows_df["label"].isin(["seizure", "background"])].copy()
    selected = []
    selection_id = f"balanced_{plan.background_to_seizure_ratio:g}"
    if plan.min_background_seizure_distance_s > 0:
        selection_id += f"_minbgdist{plan.min_background_seizure_distance_s:g}s"
    if plan.min_background_recording_edge_distance_s > 0:
        selection_id += (
            f"_minedgedist{plan.min_background_recording_edge_distance_s:g}s"
        )
    selection_id += f"_seed{plan.seed}"

    for subject_key, group in df.groupby("subject_key", sort=True):
        group = _sort_windows(group)
        seizures = group[group["label"] == "seizure"]
        background = group[group["label"] == "background"]
        background = _filter_background_by_seizure_distance(background, plan)
        background = _filter_background_by_recording_edge_distance(background, plan)
        n_seizures = len(seizures)
        if n_seizures == 0:
            logger.warning(
                "Subject %s has no seizure windows; excluded from the balanced "
                "reference for this split",
                subject_key,
            )
            continue

        target_background = int(math.ceil(plan.background_to_seizure_ratio * n_seizures))
        target_background = min(target_background, len(background))
        if target_background > 0:
            rng = np.random.RandomState(
                _stable_seed(plan.seed, "background", subject_key)
            )
            order = rng.permutation(len(background))
            background_selected = background.iloc[order[:target_background]]
        else:
            background_selected = background.iloc[0:0]

        selected.append(seizures)
        selected.append(background_selected)

    if not selected:
        return pd.DataFrame(columns=columns)

    out = (
        pd.concat(selected, axis=0)
        .loc[:, ["subject_key", "window_id", "label"]]
        .drop_duplicates(subset=["window_id"])
        .sort_values(["subject_key", "label", "window_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    out.insert(0, "selection_id", selection_id)
    out["role"] = "candidate"
    out["selected_order"] = np.arange(len(out), dtype=int)
    return out[columns]


def _filter_background_by_seizure_distance(
    background_df: pd.DataFrame,
    plan: CVPlan,
) -> pd.DataFrame:
    min_distance = float(plan.min_background_seizure_distance_s)
    if min_distance <= 0 or background_df.empty:
        return background_df

    distance_col = "nearest_seizure_distance_s"
    if distance_col not in background_df.columns:
        raise ValueError(
            "min_background_seizure_distance_s requires windows_df to include "
            "'nearest_seizure_distance_s'. Rebuild windows with "
            "build_windowed_segments_dataframe before creating folds."
        )

    distances = pd.to_numeric(background_df[distance_col], errors="coerce")
    return background_df.loc[distances >= min_distance]


def _filter_background_by_recording_edge_distance(
    background_df: pd.DataFrame,
    plan: CVPlan,
) -> pd.DataFrame:
    min_distance = float(plan.min_background_recording_edge_distance_s)
    if min_distance <= 0 or background_df.empty:
        return background_df

    distance_col = "nearest_recording_edge_distance_s"
    if distance_col not in background_df.columns:
        raise ValueError(
            "min_background_recording_edge_distance_s requires windows_df to "
            "include 'nearest_recording_edge_distance_s'. Rebuild windows with "
            "build_windowed_segments_dataframe before creating folds."
        )

    distances = pd.to_numeric(background_df[distance_col], errors="coerce")
    return background_df.loc[distances >= min_distance]


def resolve_fold_windows(
    cv_boundaries_df: pd.DataFrame,
    windows_df: pd.DataFrame,
    balanced_reference_df: pd.DataFrame,
    plan: Union[CVPlan, dict],
    evaluation_windows_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve CV boundaries into exact train/val/test window rows."""

    plan = _coerce_plan(plan)
    _validate_plan(plan)

    if cv_boundaries_df is None or cv_boundaries_df.empty:
        empty = _empty_fold_windows_df(windows_df)
        return empty, _summarise_fold_windows(empty, plan)

    _require_columns(cv_boundaries_df, ["fold_id", "pool", "recording_id"])
    _require_columns(windows_df, ["recording_id", "window_id", "label"])

    _require_columns(balanced_reference_df, ["fold_id", "pool", "window_id"])
    if plan.eval_strategy == "complete_recordings":
        if evaluation_windows_df is None:
            raise ValueError("complete_recordings requires evaluation_windows_df")
        _validate_evaluation_coverage(evaluation_windows_df)

    resolved_parts = []
    fold_ids = list(pd.unique(cv_boundaries_df["fold_id"]))
    for fold_id in fold_ids:
        boundary = cv_boundaries_df[cv_boundaries_df["fold_id"] == fold_id]
        for split, pool in [("train", "train_pool"), ("val", "val_pool"), ("test", "test_pool")]:
            if split == "val" and not _uses_validation(plan):
                continue
            recording_ids = boundary.loc[boundary["pool"] == pool, "recording_id"]
            continuous = split != "train" and plan.eval_strategy == "complete_recordings"
            source = evaluation_windows_df if continuous else windows_df
            rows = _windows_for_recordings(source, recording_ids)
            if not continuous:
                reference = balanced_reference_df[
                    (balanced_reference_df["fold_id"] == fold_id)
                    & (balanced_reference_df["pool"] == pool)
                ]
                rows = _intersect_window_ids(rows, set(reference["window_id"]))
            if rows.empty:
                raise ValueError(
                    f"Fold {fold_id}: {split} has no usable windows after selection. "
                    "Check recording eligibility, seizure availability, and sampling settings."
                )
            row_source = "fixed_train" if split == "train" else plan.eval_strategy
            resolved_parts.append(
                _add_split_columns(rows, fold_id, split, row_source, split == "train", plan)
            )

    if not resolved_parts:
        fold_windows_df = _empty_fold_windows_df(windows_df)
    else:
        fold_windows_df = pd.concat(resolved_parts, axis=0).reset_index(drop=True)
        fold_windows_df = _sort_fold_windows(fold_windows_df)

    return fold_windows_df, _summarise_fold_windows(fold_windows_df, plan)


def fold_windows_to_split_dict(
    fold_windows_df: pd.DataFrame,
) -> Dict[str, Dict[str, Optional[pd.DataFrame]]]:
    """Convert the long fold-window dataframe into fold-specific split dataframes."""

    if fold_windows_df is None or fold_windows_df.empty:
        return {}

    folds: Dict[str, Dict[str, Optional[pd.DataFrame]]] = {}
    for fold_id in pd.unique(fold_windows_df["fold_id"]):
        fold_df = fold_windows_df[fold_windows_df["fold_id"] == fold_id]
        split_dict: Dict[str, Optional[pd.DataFrame]] = {}
        for split in ["train", "val", "test"]:
            split_df = fold_df[fold_df["split"] == split].copy().reset_index(drop=True)
            split_dict[split] = split_df if not split_df.empty or split != "val" else None
        folds[str(fold_id)] = split_dict
    return folds


def _assign_lopo(recordings_df: pd.DataFrame, plan: CVPlan) -> pd.DataFrame:
    rows = []
    subjects = sorted(recordings_df["subject_key"].astype(str).unique())
    for subject_key in subjects:
        fold_id = f"subject_{_safe_token(subject_key)}"
        for rec in recordings_df.itertuples(index=False):
            pool = "test_pool" if str(rec.subject_key) == subject_key else "train_pool"
            rows.append(_boundary_row(fold_id, pool, plan.strategy, rec, "all"))
    return pd.DataFrame(rows)


def _assign_kfold_patient(recordings_df: pd.DataFrame, plan: CVPlan) -> pd.DataFrame:
    subjects = sorted(recordings_df["subject_key"].astype(str).unique())
    if plan.k <= 1:
        raise ValueError("kfold_patient requires k >= 2")
    if plan.k > len(subjects):
        raise ValueError(f"k={plan.k} is larger than the number of subjects ({len(subjects)})")

    rng = np.random.RandomState(plan.seed)
    shuffled = [subjects[i] for i in rng.permutation(len(subjects))]
    subject_folds = [set(shuffled[i::plan.k]) for i in range(plan.k)]

    rows = []
    for fold_index, test_subjects in enumerate(subject_folds):
        fold_id = f"fold_{fold_index + 1:02d}"
        for rec in recordings_df.itertuples(index=False):
            pool = "test_pool" if str(rec.subject_key) in test_subjects else "train_pool"
            rows.append(_boundary_row(fold_id, pool, plan.strategy, rec, "all"))
    return pd.DataFrame(rows)


def _assign_leave_one_centre_out(recordings_df: pd.DataFrame, plan: CVPlan) -> pd.DataFrame:
    centres = sorted(recordings_df["centre"].astype(str).unique())
    if len(centres) <= 1:
        raise ValueError("leave_one_centre_out requires at least two centres")

    rows = []
    for centre in centres:
        fold_id = f"centre_{_safe_token(centre)}"
        for rec in recordings_df.itertuples(index=False):
            pool = "test_pool" if str(rec.centre) == centre else "train_pool"
            rows.append(_boundary_row(fold_id, pool, plan.strategy, rec, "all"))
    return pd.DataFrame(rows)


def _assign_patient_specific(recordings_df: pd.DataFrame, plan: CVPlan) -> pd.DataFrame:
    """Choose one ordered split with usable seizure data in every enabled block.

    Percentages are targets. Clamp each boundary to keep a seizure in its
    held-out block and leave usable seizure data in all preceding blocks.
    Eligibility counts distinct recordings, not seizure events or windows.
    """
    _require_columns(recordings_df, ["has_seizure"])
    rows = []
    uses_val = _uses_validation(plan)
    required = 3 if uses_val else 2

    for subject_key, group in recordings_df.groupby("subject_key", sort=True):
        group = _sort_recordings(group).reset_index(drop=True)
        n_recordings = len(group)
        training_seizures = group.get("has_training_seizure", group["has_seizure"])
        training_seizures = training_seizures.fillna(False).astype(bool)
        evaluation_seizures = (
            training_seizures if plan.eval_strategy == "balanced_windows"
            else group["has_seizure"].fillna(False).astype(bool)
        )
        seizure_positions = np.flatnonzero(evaluation_seizures.to_numpy())
        training_positions = np.flatnonzero(training_seizures.to_numpy())
        if len(seizure_positions) < required:
            logger.warning(
                "Subject %s has %d usable seizure-containing recording(s); "
                "patient_specific requires at least %d. Excluded.",
                subject_key, len(seizure_positions), required,
            )
            continue

        later_seizures = (
            seizure_positions[seizure_positions > training_positions[0]]
            if len(training_positions) else []
        )
        if len(later_seizures) < required - 1:
            logger.warning(
                "Subject %s has no recording with full seizure training windows "
                "before %d later seizure-containing recording(s). Excluded.",
                subject_key, required - 1,
            )
            continue

        # Leave the first usable training seizure (and a later validation
        # seizure when needed) before test. Test includes the latest seizure.
        earliest_test = later_seizures[0] + 1 if uses_val else training_positions[0] + 1
        test_start = n_recordings - int(math.ceil(plan.test_fraction * n_recordings))
        test_start = min(max(test_start, earliest_test), seizure_positions[-1])

        if uses_val:
            latest_val = seizure_positions[seizure_positions < test_start][-1]
            # Validation starts after the first training seizure and at or
            # before the latest seizure remaining on the pre-test side.
            train_end = test_start - int(math.ceil(plan.val_fraction * test_start))
            train_end = min(max(train_end, training_positions[0] + 1), latest_val)
        else:
            train_end = test_start

        fold_id = f"patient_{_safe_token(subject_key)}"
        for pos, rec in enumerate(group.itertuples(index=False)):
            if pos < train_end:
                pool = "train_pool"
                time_block = "early"
            elif pos < test_start:
                pool = "val_pool"
                time_block = "middle"
            else:
                pool = "test_pool"
                time_block = "late"
            rows.append(_boundary_row(fold_id, pool, plan.strategy, rec, time_block))

    if not rows:
        raise ValueError(
            f"patient_specific requires at least {required} usable seizure-containing "
            "recordings per patient, with full seizure training windows before "
            "the held-out blocks"
        )
    return pd.DataFrame(rows)


def _boundary_row(fold_id: str, pool: str, strategy: str, rec, time_block: str) -> dict:
    return {
        "fold_id": fold_id,
        "pool": pool,
        "strategy": strategy,
        "subject_key": str(rec.subject_key),
        "centre": str(rec.centre),
        "recording_id": str(rec.recording_id),
        "time_block": time_block,
    }


def _assign_validation_patients(boundaries: pd.DataFrame, plan: CVPlan) -> pd.DataFrame:
    """Hold out whole patients from each fold's non-test pool before sampling."""
    if not _uses_validation(plan):
        return boundaries
    out = boundaries.copy()
    for fold_id, group in out.groupby("fold_id", sort=True):
        patients = sorted(group.loc[group["pool"] == "train_pool", "subject_key"].unique())
        if len(patients) < 2:
            raise ValueError(
                f"Fold {fold_id}: patient-level validation needs at least two non-test patients "
                "to keep training and validation separate."
            )
        n_val = min(len(patients) - 1, max(1, int(math.ceil(plan.val_fraction * len(patients)))))
        rng = np.random.RandomState(_stable_seed(plan.seed, "validation", fold_id))
        chosen = {patients[i] for i in rng.permutation(len(patients))[:n_val]}
        mask = (out["fold_id"] == fold_id) & (out["pool"] == "train_pool")
        mask &= out["subject_key"].isin(chosen)
        out.loc[mask, "pool"] = "val_pool"
    return out


def _validate_evaluation_coverage(windows_df: pd.DataFrame) -> None:
    """Reject gaps, out-of-bounds windows, and malformed partial-window metadata."""
    _require_columns(windows_df, [
        "recording_id", "recordingDuration", "start_s", "end_s",
        "valid_duration_s", "window_length_s", "seizure_duration_s", "confidence_seizure",
    ])
    for recording_id, group in windows_df.groupby("recording_id", sort=False):
        group = group.sort_values("start_s")
        starts = group["start_s"].to_numpy(dtype=float)
        ends = group["end_s"].to_numpy(dtype=float)
        durations = group["recordingDuration"].to_numpy(dtype=float)
        lengths = group["window_length_s"].to_numpy(dtype=float)
        valid = group["valid_duration_s"].to_numpy(dtype=float)
        seizure = group["seizure_duration_s"].to_numpy(dtype=float)
        coverage = group["confidence_seizure"].to_numpy(dtype=float)
        finite = all(np.isfinite(a).all() for a in [starts, ends, durations, lengths, valid, seizure, coverage])
        bounds = (
            (starts >= 0).all() and (ends > starts).all() and (ends <= durations + 1e-6).all()
            and (lengths > 0).all() and (valid <= lengths + 1e-6).all()
            and (seizure >= 0).all() and (seizure <= valid + 1e-6).all()
        )
        complete = (
            np.isclose(starts[0], 0, atol=1e-6, rtol=0)
            and np.allclose(durations, durations[0], atol=1e-6, rtol=0)
            and np.isclose(ends[-1], durations[0], atol=1e-6, rtol=0)
            and (starts[1:] <= np.maximum.accumulate(ends)[:-1] + 1e-6).all()
        )
        grid = (
            np.allclose(lengths, lengths[0], atol=1e-6, rtol=0)
            and np.allclose(valid, ends - starts, atol=1e-6, rtol=0)
            and np.allclose(valid[:-1], lengths[:-1], atol=1e-6, rtol=0)
            and np.allclose(coverage * valid, seizure, atol=1e-6, rtol=0)
        )
        if len(starts) > 1:
            steps = np.diff(starts)
            grid = grid and (steps > 0).all() and np.allclose(steps, steps[0], atol=1e-6, rtol=0)
        if not (finite and bounds and complete and grid):
            raise ValueError(f"Recording {recording_id}: evaluation windows must form a complete regular grid")


def _windows_for_recordings(windows_df: pd.DataFrame, recording_ids: Iterable[str]) -> pd.DataFrame:
    recording_ids = set(str(v) for v in recording_ids)
    if not recording_ids:
        return windows_df.iloc[0:0].copy()
    return windows_df[windows_df["recording_id"].astype(str).isin(recording_ids)].copy()


def _intersect_window_ids(windows_df: pd.DataFrame, selected_ids: set) -> pd.DataFrame:
    if windows_df is None or windows_df.empty:
        return windows_df
    if not selected_ids:
        return windows_df.iloc[0:0].copy()
    return windows_df[windows_df["window_id"].astype(str).isin(selected_ids)].copy()


def _add_split_columns(
    df: pd.DataFrame,
    fold_id: str,
    split: str,
    row_source: str,
    augment_allowed: bool,
    plan: CVPlan,
) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "augment_allowed", bool(augment_allowed))
    out.insert(0, "row_source", row_source)
    out.insert(0, "split", split)
    out.insert(0, "fold_id", fold_id)
    out["strategy"] = plan.strategy
    out["eval_strategy"] = plan.eval_strategy
    out["validation"] = plan.validation
    return out


def _summarise_fold_windows(fold_windows_df: pd.DataFrame, plan: CVPlan) -> pd.DataFrame:
    columns = [
        "fold_id",
        "split",
        "strategy",
        "eval_strategy",
        "validation",
        "n_subjects",
        "n_recordings",
        "n_windows",
        "n_background",
        "n_seizure",
        "n_mixed",
        "n_partial",
    ]
    if fold_windows_df is None or fold_windows_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (fold_id, split), group in fold_windows_df.groupby(["fold_id", "split"], sort=False):
        rows.append(
            {
                "fold_id": fold_id,
                "split": split,
                "strategy": plan.strategy,
                "eval_strategy": plan.eval_strategy,
                "validation": plan.validation,
                "n_subjects": int(group["subject_key"].nunique())
                if "subject_key" in group.columns
                else 0,
                "n_recordings": int(group["recording_id"].nunique())
                if "recording_id" in group.columns
                else 0,
                "n_windows": int(len(group)),
                "n_background": int((group["label"] == "background").sum())
                if "label" in group.columns
                else 0,
                "n_seizure": int((group["label"] == "seizure").sum())
                if "label" in group.columns
                else 0,
                "n_mixed": int((group["label"] == "mixed").sum()),
                "n_partial": int((group["valid_duration_s"] < group["window_length_s"] - 1e-6).sum())
                if "valid_duration_s" in group.columns and "window_length_s" in group.columns
                else 0,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _validate_plan(plan: CVPlan) -> None:
    if plan.strategy not in CV_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(CV_STRATEGIES)}")
    if plan.eval_strategy not in EVAL_STRATEGIES:
        raise ValueError(f"eval_strategy must be one of {sorted(EVAL_STRATEGIES)}")
    if not isinstance(plan.validation, bool):
        raise ValueError("validation must be a bool (True or False)")
    if _uses_validation(plan) and not (0 < float(plan.val_fraction) < 1):
        raise ValueError("val_fraction must be in (0, 1) when validation=True")
    if not (0 < float(plan.test_fraction) < 1):
        raise ValueError("test_fraction must be in (0, 1)")
    if float(plan.min_background_seizure_distance_s) < 0:
        raise ValueError("min_background_seizure_distance_s must be >= 0")
    if float(plan.min_background_recording_edge_distance_s) < 0:
        raise ValueError("min_background_recording_edge_distance_s must be >= 0")


def _coerce_plan(plan: Union[CVPlan, dict]) -> CVPlan:
    if isinstance(plan, CVPlan):
        coerced = plan
    elif isinstance(plan, dict):
        coerced = CVPlan(**plan)
    else:
        raise TypeError("plan must be a CVPlan or a dict")
    return CVPlan(
        strategy=str(coerced.strategy).lower(),
        eval_strategy=str(coerced.eval_strategy).lower(),
        validation=_coerce_validation_flag(coerced.validation),
        val_fraction=float(coerced.val_fraction),
        k=int(coerced.k),
        seed=int(coerced.seed),
        background_to_seizure_ratio=float(coerced.background_to_seizure_ratio),
        min_background_seizure_distance_s=float(coerced.min_background_seizure_distance_s),
        min_background_recording_edge_distance_s=float(
            coerced.min_background_recording_edge_distance_s
        ),
        test_fraction=float(coerced.test_fraction),
    )


def _coerce_validation_flag(value: object) -> bool:
    """Coerce the ``validation`` field to a bool.

    Accepts real booleans, and for backward compatibility the legacy strings
    ``"use_validation"``/``"none"`` as well as common truthy/falsy strings.
    Raises ``ValueError`` on anything ambiguous rather than guessing.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"use_validation", "true", "1", "yes"}:
            return True
        if text in {"none", "false", "0", "no", ""}:
            return False
    raise ValueError(
        "validation must be a bool (or a recognised legacy value such as "
        "'use_validation'/'none'); got {!r}".format(value)
    )


def _uses_validation(plan: CVPlan) -> bool:
    return bool(plan.validation)


def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _subject_key_series(df: pd.DataFrame) -> pd.Series:
    centre = df["centre"].map(_value_to_string)
    subject = df["subject"].map(_value_to_string)
    return centre + ":" + subject


def _recording_ids(recording_keys: pd.DataFrame) -> pd.Series:
    hash_columns = [c for c in CATALOG_RECORDING_COLUMNS if c in recording_keys.columns]
    hashes = _hash_rows(recording_keys, hash_columns).str.slice(0, 8)
    centre = recording_keys["centre"].map(_safe_token)
    subject = recording_keys["subject"].map(_safe_token)
    session = recording_keys["session"].map(_safe_token)
    run = recording_keys["run"].map(_safe_token)
    return "rec_" + centre + "_sub-" + subject + "_ses-" + session + "_run-" + run + "_" + hashes


def _hash_rows(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    key_df = df[list(columns)].copy()
    for col in key_df.columns:
        key_df[col] = key_df[col].map(_value_to_string)
    hashed = pd.util.hash_pandas_object(key_df, index=False).astype("uint64")
    return hashed.map(lambda value: f"{int(value):016x}")


def _stable_seed(seed: int, *parts: object) -> int:
    text = "|".join([str(seed)] + [_value_to_string(p) for p in parts])
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _value_to_string(value: object) -> str:
    if pd.isna(value):
        return "na"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return str(int(number)) if number.is_integer() else f"{number:g}"
    return str(value)


def _safe_token(value: object) -> str:
    text = _value_to_string(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "na"


def _bids_order_key(series: pd.Series) -> pd.Series:
    """Return a sort key for a BIDS entity (session or run).

    BIDS entity labels are strings such as "01", "02", or sometimes "1", "10".
    Sorting them as text is wrong ("10" would precede "2"), so when every value
    is numeric this returns a numeric key. If any value is non-numeric (e.g. a
    named session like "pre"), it falls back to the original string so ordering
    is still deterministic.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric
    return series.astype(str)


def _add_recording_order(recordings_df: pd.DataFrame) -> pd.DataFrame:
    """Assign a per-subject chronological ``recording_order`` from BIDS entities.

    Ordering uses the BIDS ``session`` then ``run`` labels (numerically when
    they are numeric), which reflect acquisition order. Acquisition timestamps
    are deliberately not used: ``dateTime`` is frequently randomised during
    pseudo-anonymisation, so it is not a reliable chronological signal.
    """
    df = recordings_df.copy()
    df = df.sort_values("subject_key", kind="mergesort").reset_index(drop=True)

    order_keys = {}
    for col in ["session", "run", "recording_id"]:
        if col in df.columns:
            order_keys[f"_order_{col}"] = _bids_order_key(df[col])
    for key, values in order_keys.items():
        df[key] = values

    sort_cols = ["subject_key"] + list(order_keys.keys())
    df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    df["recording_order"] = df.groupby("subject_key", sort=False).cumcount()
    df = df.drop(columns=list(order_keys.keys()))
    return df


def _sort_recordings(recordings_df: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = [
        "centre",
        "subject",
        "recording_order",
        "session",
        "run",
        "recording_id",
    ]
    sort_cols = [c for c in candidate_cols if c in recordings_df.columns]
    return recordings_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _sort_windows(windows_df: pd.DataFrame) -> pd.DataFrame:
    if windows_df is None:
        return windows_df
    sort_cols = [c for c in SORT_WINDOW_COLUMNS if c in windows_df.columns]
    if "window_id" in windows_df.columns:
        sort_cols.append("window_id")
    if not sort_cols:
        return windows_df.reset_index(drop=True)
    return windows_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)


def _sort_fold_windows(fold_windows_df: pd.DataFrame) -> pd.DataFrame:
    split_order = {"train": 0, "val": 1, "test": 2}
    df = fold_windows_df.copy()
    df["_split_order"] = df["split"].map(split_order).fillna(99).astype(int)
    sort_cols = ["fold_id", "_split_order"] + [c for c in SORT_WINDOW_COLUMNS if c in df.columns]
    if "window_id" in df.columns:
        sort_cols.append("window_id")
    df = df.sort_values(sort_cols, kind="mergesort").drop(columns=["_split_order"])
    return df.reset_index(drop=True)


def _empty_boundaries_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "fold_id",
            "pool",
            "strategy",
            "subject_key",
            "centre",
            "recording_id",
            "time_block",
        ]
    )


def _empty_fold_windows_df(windows_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    base_columns = list(windows_df.columns) if windows_df is not None else []
    columns = ["fold_id", "split", "row_source", "augment_allowed"] + base_columns + [
        "strategy",
        "eval_strategy",
        "validation",
    ]
    return pd.DataFrame(columns=list(dict.fromkeys(columns)))
