"""Metadata-only segment and window table construction for EEG-BIDS datasets.

This module turns a BIDS-like EEG dataset (raw or preprocessed) into
deterministic, inspectable pandas tables. It never loads EEG signal arrays;
it reads only ``events.tsv`` files and lightweight duration metadata.

The pipeline runs in two stages, in this order:

1. ``build_segments_dataframe`` walks a dataset root, and for each recording
   parses its seizure events and total duration, then expands the timeline into
   contiguous, non-overlapping ``background`` and ``seizure`` intervals. Each
   row of ``segments_df`` is one such interval.

2. ``build_windowed_segments_dataframe`` slices those intervals into
   fixed-length windows with configurable overlap. Because windows are cut
   strictly inside a single interval, every window is wholly ``seizure`` or
   wholly ``background`` -- there is no mixed-label window. Each row of
   ``windows_df`` is one model-ready sample instruction.

``build_evaluation_windows_dataframe`` builds a separate regular grid covering
each complete recording, including mixed-label windows and the final partial
window. Training and evaluation share recording metadata, but their window
placement serves different purposes.

``build_segments_dataframe_multi`` runs stage 1 across several roots after
checking that their signal metadata (sampling rate, channel names) agree.

The downstream fold module consumes ``windows_df``; the column contract it
relies on is declared in ``WINDOW_COLUMNS`` below.
"""

import json
import math
import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
import h5py

logger = logging.getLogger(__name__)

__all__ = [
    "build_segments_dataframe",
    "build_segments_dataframe_multi",
    "build_windowed_segments_dataframe",
    "build_evaluation_windows_dataframe",
    "SEGMENT_COLUMNS",
    "WINDOW_COLUMNS",
]

# Column contract for the output tables. WINDOW_COLUMNS is the interface the
# fold module depends on; keeping it explicit here means the contract is visible
# in one place rather than reconstructed from the row-building code.
SEGMENT_COLUMNS = [
    "dataset_root",
    "centre",
    "subject",
    "session",
    "run",
    "start_s",
    "end_s",
    "label",
    "channels",
    "dateTime",
    "recordingDuration",
    "events_path",
    "signal_path",
    "signal_format",
    "edf_path",
]

WINDOW_COLUMNS = [
    "dataset_root",
    "centre",
    "subject",
    "session",
    "run",
    "start_s",
    "end_s",
    "label",
    "channels",
    "confidence_seizure",
    "nearest_seizure_distance_s",
    "nearest_recording_edge_distance_s",
    "dateTime",
    "recordingDuration",
    "events_path",
    "signal_path",
    "signal_format",
    "edf_path",
    "valid_duration_s",
    "window_length_s",
    "seizure_duration_s",
]


@dataclass
class EventRow:
    """One row parsed from a BIDS ``events.tsv`` file."""

    onset: float
    duration: float
    label: str
    channels: Optional[str]


def _parse_events_tsv(events_path: str) -> Tuple[List[EventRow], Optional[float]]:
    """
    Parse a BIDS-style events.tsv file.

    Expected columns (best-effort, tolerant to naming):
      - onset (seconds)
      - duration (seconds)
      - eventType or trial_type (label)
      - channels (comma-separated) [optional]
      - recordingDuration (may exist as a column)

    Returns:
      events: list of EventRow (seizures and any other events present)
      recording_duration: float or None (if found either in the TSV column or inferred later)
    """
    if not os.path.exists(events_path):
        return [], None

    # Read with pandas, be forgiving with missing values
    df = pd.read_csv(events_path, sep="\t")

    # Normalize column names for robustness
    cols = {c.lower(): c for c in df.columns}
    onset_col = cols.get("onset")
    duration_col = cols.get("duration")
    label_col = cols.get("eventtype") or cols.get("trial_type")
    channels_col = cols.get("channels")
    recdur_col = cols.get("recordingduration") or cols.get("recording_duration")

    events: List[EventRow] = []
    if onset_col and duration_col:
        for _, row in df.iterrows():
            onset = float(row[onset_col]) if pd.notna(row[onset_col]) else 0.0
            duration = float(row[duration_col]) if pd.notna(row[duration_col]) else 0.0
            label = str(row[label_col]) if label_col and pd.notna(row.get(label_col, None)) else "event"
            channels = str(row[channels_col]) if channels_col and pd.notna(row.get(channels_col, None)) else None
            events.append(EventRow(onset=onset, duration=duration, label=label, channels=channels))

    rec_dur: Optional[float] = None
    if recdur_col and recdur_col in df.columns and pd.notna(df.loc[0, recdur_col]):
        rec_dur = float(df.loc[0, recdur_col])

    return events, rec_dur


def _parse_events_metadata(events_path: str) -> dict:
    """Read recording-level metadata available in an events.tsv file."""
    meta = {
        "dateTime": None,
        "recordingDuration": None,
    }
    if not os.path.exists(events_path):
        return meta
    try:
        df = pd.read_csv(events_path, sep="\t")
    except Exception:
        return meta
    if df.empty:
        return meta

    cols = {c.lower(): c for c in df.columns}
    datetime_col = cols.get("datetime") or cols.get("date_time")
    recdur_col = cols.get("recordingduration") or cols.get("recording_duration")

    if datetime_col and datetime_col in df.columns:
        values = df[datetime_col].dropna()
        if not values.empty:
            meta["dateTime"] = str(values.iloc[0])
    if recdur_col and recdur_col in df.columns:
        values = df[recdur_col].dropna()
        if not values.empty:
            try:
                meta["recordingDuration"] = float(values.iloc[0])
            except Exception:
                meta["recordingDuration"] = None
    return meta


def _read_recording_duration_from_sidecar(json_path: str) -> Optional[float]:
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path, "r") as f:
            meta = json.load(f)
        # Common BIDS field name
        dur = meta.get("RecordingDuration")
        if isinstance(dur, (int, float)):
            return float(dur)
    except Exception:
        return None
    return None


def _resolve_recording_duration(
    primary: Optional[float],
    sidecar_json: Optional[str],
    signal_path: Optional[str],
) -> Optional[float]:
    """Resolve a recording's duration from the first available source.

    Preference order: the ``primary`` value already parsed from the events.tsv,
    then the BIDS JSON sidecar, then the H5 signal file. Returns ``None`` when
    none of the sources provide a usable duration.
    """
    if primary:
        return primary
    from_sidecar = (
        _read_recording_duration_from_sidecar(sidecar_json) if sidecar_json else None
    )
    if from_sidecar:
        return from_sidecar
    return _read_recording_duration_from_h5(signal_path)


def _read_recording_duration_from_h5(h5_path: Optional[str]) -> Optional[float]:
    if not h5_path or not os.path.exists(h5_path):
        return None
    try:
        with h5py.File(h5_path, "r") as h5:
            if "duration_s" in h5.attrs:
                return float(h5.attrs["duration_s"])
            if "metadata_json" in h5:
                raw_json = h5["metadata_json"][()]
                if isinstance(raw_json, bytes):
                    raw_json = raw_json.decode("utf-8")
                meta = json.loads(str(raw_json))
                dur = meta.get("RecordingDuration")
                if isinstance(dur, (int, float)):
                    return float(dur)
    except Exception:
        return None
    return None


def _signal_format_from_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    suffix = os.path.splitext(path)[1].lower()
    if suffix == ".edf":
        return "edf"
    if suffix in {".h5", ".hdf5"}:
        return "h5"
    return suffix.lstrip(".") or None


def _find_signal_file_for_events(events_path: str) -> Optional[str]:
    """Find the EEG signal file corresponding to an events.tsv path."""
    dirpath = os.path.dirname(events_path)
    base_no_events = os.path.basename(events_path).replace("_events.tsv", "")
    candidates = [
        os.path.join(dirpath, base_no_events + "_eeg_desc-preprocessed.h5"),
        os.path.join(dirpath, base_no_events + "_eeg_desc-preprocessed.edf"),
        os.path.join(dirpath, base_no_events + "_eeg.h5"),
        os.path.join(dirpath, base_no_events + "_eeg.edf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    for fname in sorted(os.listdir(dirpath)):
        if fname.lower().endswith((".h5", ".hdf5", ".edf")) and base_no_events in fname:
            return os.path.join(dirpath, fname)
    return None


def _find_eeg_file_for_events(events_path: str) -> Optional[str]:
    """Backward-compatible alias for the recording signal file path."""
    return _find_signal_file_for_events(events_path)


def _infer_ids_from_path(path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Infer subject/session/run IDs from a typical BIDS path.

    Returns (subject_id, session_id, run_id)
    """
    # Look for tokens like sub-xx, ses-yy, run-zz in the path
    base = os.path.basename(path)
    parts = path.split(os.sep)
    subject = None
    session = None
    run = None
    for token in parts + [base]:
        if token.startswith("sub-"):
            subject = token[4:]
        elif token.startswith("ses-"):
            session = token[4:]
        elif token.startswith("run-"):
            run = token[4:]
    # Also try filename pattern e.g. sub-01_ses-01_task-..._run-01_events.tsv
    for frag in base.split("_"):
        if frag.startswith("sub-"):
            subject = frag[4:]
        elif frag.startswith("ses-"):
            session = frag[4:]
        elif frag.startswith("run-"):
            run = frag[4:]
    return subject, session, run


def _infer_centre_from_path(path: str) -> Optional[str]:
    """Infer a centre name from the dataset root path (e.g., 'tusz', 'siena', 'chbmit')."""
    parts = path.strip(os.sep).split(os.sep)
    if not parts:
        return None
    # Heuristic: take the last part before typical BIDS tokens if present
    for token in reversed(parts):
        if token.lower() in {"tusz", "siena", "chbmit"}:
            return token.lower()
    # Fallback to the last directory name
    return parts[-1].lower()


def _iter_events_files(dataset_root: str):
    """Yield events.tsv paths under a BIDS-like root, skipping derivatives for raw roots."""
    root_norm = os.path.abspath(dataset_root)
    is_in_derivatives = f"{os.sep}derivatives{os.sep}" in root_norm
    for dirpath, dirnames, filenames in os.walk(dataset_root):
        dirnames.sort()
        if not is_in_derivatives and f"{os.sep}derivatives{os.sep}" in os.path.abspath(dirpath):
            continue
        for fname in sorted(filenames):
            if fname.endswith("_events.tsv"):
                yield os.path.join(dirpath, fname)


def _read_edf_metadata(edf_path: str) -> Optional[dict]:
    """Read EDF file metadata and extract parameters for consistency checking.

    Only returns:
      - sampling_frequencies: sorted unique list of channel sample rates
      - channel_names: sorted list of signal labels
    """
    try:
        import pyedflib
        with pyedflib.EdfReader(edf_path) as f:
            # Get sampling frequencies
            sample_freqs = f.getSampleFrequencies()
            # Get channel names
            channel_names = f.getSignalLabels()
            metadata = {
                "sampling_frequencies": sorted(set(sample_freqs)),
                "channel_names": sorted(channel_names),
            }
            return metadata
    except ImportError:
        logger.warning("pyedflib not available; cannot read EDF metadata. Install with: pip install pyedflib")
        return None
    except Exception as e:
        logger.warning("Could not read EDF file %s: %s", edf_path, e)
        return None


def _read_h5_metadata(h5_path: str) -> Optional[dict]:
    """Read H5 signal metadata for consistency checking."""
    try:
        with h5py.File(h5_path, "r") as h5:
            if "signals" not in h5 or "channel_names" not in h5:
                return None
            sfreq = float(h5.attrs["sampling_frequency"])
            channel_names = [
                ch.decode("utf-8") if isinstance(ch, bytes) else str(ch)
                for ch in h5["channel_names"][()]
            ]
            return {
                "sampling_frequencies": [sfreq],
                "channel_names": sorted(channel_names),
            }
    except Exception as e:
        logger.warning("Could not read H5 file %s: %s", h5_path, e)
        return None


def _read_signal_metadata(signal_path: str) -> Optional[dict]:
    suffix = os.path.splitext(signal_path)[1].lower()
    if suffix == ".edf":
        return _read_edf_metadata(signal_path)
    if suffix in {".h5", ".hdf5"}:
        return _read_h5_metadata(signal_path)
    return None


def _find_sample_signal_file(dataset_root: str) -> Optional[str]:
    """Find a representative signal file for metadata extraction.

    Behavior:
      - If the provided root is inside a derivatives folder, allow any signal under it
      - Otherwise, skip any signal files inside directories containing 'derivatives'
    """
    root_norm = os.path.abspath(dataset_root)
    is_in_derivatives = "{}derivatives{}".format(os.sep, os.sep) in root_norm
    candidates: List[str] = []
    for dirpath, _, filenames in os.walk(dataset_root):
        # Skip derivatives when scanning a raw dataset root
        if not is_in_derivatives and ("{}derivatives{}".format(os.sep, os.sep) in os.path.abspath(dirpath)):
            continue
        for fname in sorted(filenames):
            if fname.lower().endswith((".h5", ".hdf5", ".edf")):
                candidates.append(os.path.join(dirpath, fname))
    return sorted(candidates)[0] if candidates else None


def _find_sample_edf_file(dataset_root: str) -> Optional[str]:
    """Backward-compatible wrapper for representative signal discovery."""
    return _find_sample_signal_file(dataset_root)


def _collect_preprocessing_signatures(dataset_roots: List[str]) -> Tuple[bool, List[Tuple[str, Optional[dict]]]]:
    """
    Check signal metadata consistency across dataset roots using a sample file per root.
    Returns (all_equal, details), where details is a list of (root, metadata_dict or None).
    Compares only sampling frequencies and channel names.
    """
    results: List[Tuple[str, Optional[dict]]] = []
    canonical: Optional[dict] = None
    all_equal = True
    
    for root in dataset_roots:
        sample_edf = _find_sample_signal_file(root)
        found_metadata: Optional[dict] = None
        if sample_edf:
            found_metadata = _read_signal_metadata(sample_edf)
        
        results.append((root, found_metadata))
        if found_metadata is not None:
            if canonical is None:
                canonical = found_metadata
            elif found_metadata != canonical:
                all_equal = False
    
    return all_equal, results


def _merge_seizure_intervals(
    seizure_intervals: List[Tuple[float, float, Optional[str]]],
) -> List[Tuple[float, float, Optional[str]]]:
    """Merge overlapping or adjacent seizure intervals into disjoint spans.

    Intervals are ``(start_s, end_s, channels)``. Two intervals are merged when
    the next one starts at or before the current one ends. When merged spans
    carry different channel strings, the channels are unioned into a single
    sorted, comma-separated string. The input list is not modified.
    """
    if not seizure_intervals:
        return []

    ordered = sorted(seizure_intervals, key=lambda x: x[0])
    merged: List[Tuple[float, float, Optional[str]]] = []
    cur_s, cur_e, cur_ch = ordered[0]
    for s, e, ch in ordered[1:]:
        if s <= cur_e:  # overlap or adjacent
            cur_e = max(cur_e, e)
            if cur_ch and ch and cur_ch != ch:
                parts = [c.strip() for c in (cur_ch + "," + ch).split(",") if c.strip()]
                cur_ch = ",".join(sorted(set(parts)))
            elif not cur_ch:
                cur_ch = ch
        else:
            merged.append((cur_s, cur_e, cur_ch))
            cur_s, cur_e, cur_ch = s, e, ch
    merged.append((cur_s, cur_e, cur_ch))
    return merged


def _expand_segments(recording_duration: float, events: List[EventRow], seizure_label_prefix: str = "sz") -> List[Tuple[float, float, str, Optional[str]]]:
    """
    Expand timeline into contiguous non-overlapping segments labeled 'background' or seizure label.

    Assumptions:
      - Events can include various labels; any label starting with `seizure_label_prefix` is treated as seizure.
      - Overlapping seizure events will be merged into seizure spans.
    Returns list of tuples: (start_s, end_s, label, channels)
    """
    if recording_duration <= 0:
        return []

    # Build seizure intervals, clipped to the recording bounds
    seizure_intervals: List[Tuple[float, float, Optional[str]]] = []
    for e in events:
        is_seizure = e.label.lower().startswith(seizure_label_prefix)
        if is_seizure and e.duration > 0:
            start = max(0.0, e.onset)
            end = min(recording_duration, e.onset + e.duration)
            if end > start:
                seizure_intervals.append((start, end, e.channels))

    seizure_intervals = _merge_seizure_intervals(seizure_intervals)

    # Create background and seizure segments spanning the whole recording
    segments: List[Tuple[float, float, str, Optional[str]]] = []
    cursor = 0.0
    for s, e, ch in seizure_intervals:
        if s > cursor:
            segments.append((cursor, s, "background", None))
        segments.append((s, e, "seizure", ch))
        cursor = e
    if cursor < recording_duration:
        segments.append((cursor, recording_duration, "background", None))

    return segments


def _segment_row(
    dataset_root: str,
    centre: Optional[str],
    subject: Optional[str],
    session: Optional[str],
    run: Optional[str],
    start_s: float,
    end_s: float,
    label: str,
    channels: Optional[str],
    rec_dur: float,
    date_time: Optional[str],
    events_path: str,
    signal_path: Optional[str],
) -> dict:
    """Assemble one segment row. Centralises the segment schema so every builder
    that emits base segments produces the exact same columns.
    """
    return {
        "dataset_root": dataset_root,
        "centre": centre,
        "subject": subject,
        "session": session,
        "run": run,
        "start_s": float(start_s),
        "end_s": float(end_s),
        "label": label,
        "channels": channels,
        "dateTime": date_time,
        "recordingDuration": float(rec_dur),
        "events_path": events_path,
        "signal_path": signal_path,
        "signal_format": _signal_format_from_path(signal_path),
        "edf_path": signal_path,
    }


def build_segments_dataframe(
    dataset_root: str,
    save_per_recording: bool = False,
    per_recording_format: str = "csv",
    centre: Optional[str] = None,
) -> pd.DataFrame:
    """
    Walk a dataset root (original or preprocessed) and build a dataframe of segments.

    Output columns:
      - dataset_root, subject, session, run
      - start_s, end_s, label, channels
    """
    rows: List[dict] = []
    centre_name = centre or _infer_centre_from_path(dataset_root)

    for events_path in _iter_events_files(dataset_root):
            dirpath = os.path.dirname(events_path)
            fname = os.path.basename(events_path)

            # Sidecar JSON: prefer preprocessed or raw sidecar (used only to read duration)
            base_no_events = fname.replace("_events.tsv", "")
            sidecar_candidates = [
                os.path.join(dirpath, base_no_events + "_eeg_desc-preprocessed.json"),
                os.path.join(dirpath, base_no_events + "_eeg.json"),
            ]
            sidecar_json = next((p for p in sidecar_candidates if os.path.exists(p)), None)
            signal_path = _find_signal_file_for_events(events_path)

            subject, session, run = _infer_ids_from_path(events_path)
            events, rec_dur_from_tsv = _parse_events_tsv(events_path)
            events_meta = _parse_events_metadata(events_path)
            rec_dur = _resolve_recording_duration(rec_dur_from_tsv, sidecar_json, signal_path)
            if not rec_dur:
                # No usable duration from TSV, sidecar, or H5: skip this
                # recording rather than emitting segments of unknown extent.
                logger.warning(
                    "Skipping recording (no recording duration found): %s", events_path
                )
                continue

            segments = _expand_segments(rec_dur, events)

            # Optionally save per recording dataframe alongside source files
            if save_per_recording and subject and session and run:
                per_rows = [
                    _segment_row(
                        dataset_root, centre_name, subject, session, run,
                        start_s, end_s, label, ch,
                        rec_dur, events_meta.get("dateTime"), events_path, signal_path,
                    )
                    for start_s, end_s, label, ch in segments
                ]
                per_df = pd.DataFrame(per_rows)
                if not per_df.empty:
                    base_no_events = fname.replace("_events.tsv", "")
                    out_suffix = "_segments.{}".format("parquet" if per_recording_format.lower() == "parquet" else "csv")
                    out_path = os.path.join(dirpath, base_no_events + out_suffix)
                    if per_recording_format.lower() == "parquet":
                        per_df.to_parquet(out_path, index=False)
                    else:
                        per_df.to_csv(out_path, index=False)

            # Collect into global dataframe
            for start_s, end_s, label, ch in segments:
                rows.append(
                    _segment_row(
                        dataset_root, centre_name, subject, session, run,
                        start_s, end_s, label, ch,
                        rec_dur, events_meta.get("dateTime"), events_path, signal_path,
                    )
                )

    df = pd.DataFrame(rows)
    # Deterministic ordering
    if not df.empty:
        df = df.sort_values(
            by=["centre", "subject", "session", "run", "start_s", "end_s"],
            kind="mergesort",
        ).reset_index(drop=True)
    return df


def build_segments_dataframe_multi(
    dataset_roots: List[str],
    save_per_recording: bool = False,
    per_recording_format: str = "csv",
) -> pd.DataFrame:
    """Build a combined segments dataframe across multiple dataset roots.

    Before building, verify that signal metadata (sampling_frequencies,
    channel_names) match across roots.
    """
    equal, details = _collect_preprocessing_signatures(dataset_roots)
    # Log a one-line summary per root so a mismatch is diagnosable without
    # dumping the full channel/frequency tables to stdout.
    for root, params in details:
        if isinstance(params, dict) and ("sampling_frequencies" in params or "channel_names" in params):
            logger.info(
                "Consistency check %s: sfreq=%s, n_channels=%s",
                root,
                params.get("sampling_frequencies"),
                len(params.get("channel_names") or []),
            )
        else:
            logger.info("Consistency check %s: signal metadata MISSING", root)
    if not equal:
        raise ValueError("Dataset metadata mismatch across datasets")

    frames: List[pd.DataFrame] = []
    for root in dataset_roots:
        frames.append(
            build_segments_dataframe(
                root,
                save_per_recording=save_per_recording,
                per_recording_format=per_recording_format,
                centre=_infer_centre_from_path(root),
            )
        )
    out = pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out = out.sort_values(
            by=["centre", "subject", "session", "run", "start_s", "end_s"], kind="mergesort"
        ).reset_index(drop=True)
    return out


def _window_row(
    context: dict,
    start_s: float,
    end_s: float,
    label: str,
    channels: Optional[str],
    confidence_seizure: float,
    nearest_seizure_distance_s: float,
    nearest_recording_edge_distance_s: float,
    window_length_s: Optional[float] = None,
) -> dict:
    """Assemble one window row from the shared recording ``context`` and the
    per-window fields. Centralises the window schema so the two window loops
    (seizure and background) cannot drift out of sync.
    """
    return {
        "dataset_root": context["dataset_root"],
        "centre": context["centre"],
        "subject": context["subject"],
        "session": context["session"],
        "run": context["run"],
        "start_s": float(start_s),
        "end_s": float(end_s),
        "label": label,
        "channels": channels,
        "confidence_seizure": confidence_seizure,
        "nearest_seizure_distance_s": nearest_seizure_distance_s,
        "nearest_recording_edge_distance_s": nearest_recording_edge_distance_s,
        "dateTime": context["dateTime"],
        "recordingDuration": float(context["recordingDuration"]),
        "events_path": context["events_path"],
        "signal_path": context["signal_path"],
        "signal_format": context["signal_format"],
        "edf_path": context["edf_path"],
        "valid_duration_s": float(end_s - start_s),
        "window_length_s": float(window_length_s if window_length_s is not None else end_s - start_s),
        "seizure_duration_s": float(confidence_seizure * (end_s - start_s)),
    }


def _recording_segments(segments_df: pd.DataFrame):
    """Yield shared recording metadata and merged seizure intervals.

    Both window builders use the full duration, including recordings shorter
    than a window. Missing optional session/run identifiers remain valid groups.
    """
    group_cols = ["dataset_root", "centre", "subject", "session", "run"]
    for group_key, group in segments_df.groupby(group_cols, sort=False, dropna=False):
        first = group.iloc[0]
        duration = first.get("recordingDuration")
        if duration is None or pd.isna(duration):
            duration = group["end_s"].max()
        duration = float(duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Recording {group_key} has invalid duration {duration}")

        context = dict(zip(group_cols, group_key))
        for col in ["dateTime", "events_path", "signal_path", "signal_format", "edf_path"]:
            context[col] = first.get(col)
        if pd.isna(context["signal_path"]):
            context["signal_path"] = context["edf_path"]
        if pd.isna(context["edf_path"]):
            context["edf_path"] = context["signal_path"]
        if pd.isna(context["signal_format"]):
            context["signal_format"] = _signal_format_from_path(context["signal_path"])
        context["recordingDuration"] = duration

        seizures = [
            (max(0.0, float(row.start_s)), min(duration, float(row.end_s)),
             row.channels if pd.notna(row.channels) else None)
            for row in group.itertuples(index=False)
            if row.label == "seizure" and min(duration, float(row.end_s)) > max(0.0, float(row.start_s))
        ]
        yield context, _merge_seizure_intervals(seizures)


def _save_windows_per_recording(
    windows_df: pd.DataFrame, per_recording_format: str, name: str = "windows"
) -> None:
    """Write one window file per recording, next to that recording's source data.

    The output path mirrors the events file: ``<base>_windows.csv`` (or
    ``.parquet``), where ``base`` is the events filename without the trailing
    ``_events.tsv``. Each recording is identified by its ``events_path`` column,
    so no dataset walk is needed -- the paths already travel with the rows.
    Recordings without a usable ``events_path`` are skipped.
    """
    if windows_df is None or windows_df.empty or "events_path" not in windows_df.columns:
        return

    fmt = per_recording_format.lower()
    suffix = f"_{name}.parquet" if fmt == "parquet" else f"_{name}.csv"
    for events_path, group in windows_df.groupby("events_path", sort=False):
        if not isinstance(events_path, str) or not events_path:
            continue
        dirpath = os.path.dirname(events_path)
        base_no_events = os.path.basename(events_path).replace("_events.tsv", "")
        out_path = os.path.join(dirpath, base_no_events + suffix)
        if fmt == "parquet":
            group.to_parquet(out_path, index=False)
        else:
            group.to_csv(out_path, index=False)


def build_windowed_segments_dataframe(
    segments_df: pd.DataFrame,
    segment_length: float,
    overlap_seizure: float = 0.0,
    overlap_background: float = 0.0,
    save_per_recording: bool = False,
    per_recording_format: str = "csv",
) -> pd.DataFrame:
    """
    Create a new dataframe of fixed-length windows with configurable overlaps.

    Windows are generated per recording (subject/session/run) using two passes:
      1) Background-aligned windows with step = segment_length - overlap_background starting at 0s
      2) Seizure-aligned windows for each seizure segment with step = segment_length - overlap_seizure starting at the seizure onset

    Because base segments are non-overlapping and windows are generated strictly
    inside seizure or background intervals, each window is either fully seizure
    (confidence_seizure = 1.0) or fully background (confidence_seizure = 0.0).
    The `label` mirrors that logic and no 'mixed' state is produced.

    Background windows also receive ``nearest_seizure_distance_s``, the gap in
    seconds from the window edge to the nearest seizure start/end in the same
    recording. Seizure windows have distance 0.0. Background windows in
    seizure-free recordings have infinite distance.
    All windows receive ``nearest_recording_edge_distance_s``, the gap in
    seconds from the window edge to the nearest recording start/end.

    When ``save_per_recording`` is True, one window file per recording is also
    written next to that recording's source data (``<base>_windows.csv`` or
    ``.parquet``, selected by ``per_recording_format``). This mirrors the
    ``save_per_recording`` option on ``build_segments_dataframe``. The returned
    dataframe is unchanged by this option.
    """
    if segments_df is None or segments_df.empty:
        return pd.DataFrame(columns=WINDOW_COLUMNS)

    if segment_length <= 0:
        raise ValueError("segment_length must be positive")
    if overlap_seizure < 0 or overlap_background < 0:
        raise ValueError("overlaps must be non-negative")
    if overlap_seizure >= segment_length or overlap_background >= segment_length:
        raise ValueError("overlap must be smaller than segment_length")

    step_bg = segment_length - overlap_background
    step_sz = segment_length - overlap_seizure
    eps = 1e-6  # numeric tolerance for floating comparisons

    rows: List[dict] = []

    for context, seizure_intervals in _recording_segments(segments_df):
        rec_start = 0.0
        rec_end = context["recordingDuration"]
        if rec_end - rec_start < segment_length:
            continue

        # Derive background intervals as complement of seizure within [rec_start, rec_end]
        background_intervals: List[Tuple[float, float]] = []
        cursor = rec_start
        for s, e, _ in seizure_intervals:
            if s > cursor:
                background_intervals.append((cursor, s))
            cursor = max(cursor, e)
        if cursor < rec_end:
            background_intervals.append((cursor, rec_end))

        # 1) Seizure windows strictly inside seizure intervals (full seizure windows)
        for s, e, ch in seizure_intervals:
            t = s
            while t + segment_length <= e + eps:
                start = round(t, 9)
                end = round(t + segment_length, 9)
                rows.append(
                    _window_row(
                        context,
                        start_s=start,
                        end_s=end,
                        label="seizure",
                        channels=ch if ch else None,
                        confidence_seizure=1.0,
                        nearest_seizure_distance_s=0.0,
                        nearest_recording_edge_distance_s=_nearest_recording_edge_distance(
                            float(start), float(end), rec_start, rec_end
                        ),
                    )
                )
                t += step_sz

        # 2) Background windows strictly inside background intervals (full background windows)
        for s, e in background_intervals:
            t = s
            while t + segment_length <= e + eps:
                start = round(t, 9)
                end = round(t + segment_length, 9)
                rows.append(
                    _window_row(
                        context,
                        start_s=start,
                        end_s=end,
                        label="background",
                        channels=None,
                        confidence_seizure=0.0,
                        nearest_seizure_distance_s=_nearest_seizure_distance(
                            float(start), float(end), seizure_intervals
                        ),
                        nearest_recording_edge_distance_s=_nearest_recording_edge_distance(
                            float(start), float(end), rec_start, rec_end
                        ),
                    )
                )
                t += step_bg

    out = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
    if not out.empty:
        out = out.sort_values(
            by=["centre", "subject", "session", "run", "start_s", "end_s"], kind="mergesort"
        ).reset_index(drop=True)
    if save_per_recording:
        _save_windows_per_recording(out, per_recording_format)
    return out


def build_evaluation_windows_dataframe(
    segments_df: pd.DataFrame,
    segment_length: float,
    stride: Optional[float] = None,
    save_per_recording: bool = False,
    per_recording_format: str = "csv",
) -> pd.DataFrame:
    """Cover each recording with a regular grid independent of seizure labels.

    ``stride`` defaults to ``segment_length`` and cannot exceed it, so no
    recording time is skipped. The final window ends at the recording end;
    ``valid_duration_s`` records the available signal and ``window_length_s``
    records the requested model input length. A loader may pad the difference,
    but padding must not contribute to scoring.

    Labels are ``background``, ``seizure``, or ``mixed``. ``confidence_seizure``
    is the fraction of valid window time annotated as seizure, not a clinical
    confidence score. ``seizure_duration_s`` preserves its duration explicitly;
    ``events_path`` links to the original intervals for event-level scoring.
    No binary scoring threshold is imposed on mixed windows.
    """
    stride = segment_length if stride is None else stride
    if not math.isfinite(segment_length) or segment_length <= 0:
        raise ValueError("segment_length must be finite and positive")
    if not math.isfinite(stride) or not 0 < stride <= segment_length:
        raise ValueError("stride must be positive and no larger than segment_length")
    if segments_df is None or segments_df.empty:
        return pd.DataFrame(columns=WINDOW_COLUMNS)

    rows = []
    for context, seizures in _recording_segments(segments_df):
        duration = context["recordingDuration"]
        index = 0
        while True:
            start = round(index * stride, 9)
            end = min(round(start + segment_length, 9), duration)
            overlaps = [(max(start, s), min(end, e), ch) for s, e, ch in seizures
                        if s < end and e > start]
            seizure_time = min(end - start, sum(e - s for s, e, _ in overlaps))
            coverage = seizure_time / (end - start)
            label = "background" if coverage == 0 else "seizure" if coverage >= 1 else "mixed"
            channels = sorted({c.strip() for _, _, ch in overlaps if ch
                               for c in ch.split(",") if c.strip()})
            rows.append(_window_row(
                context, start, end, label, ",".join(channels) or None, coverage,
                _nearest_seizure_distance(start, end, seizures),
                _nearest_recording_edge_distance(start, end, 0.0, duration),
                window_length_s=segment_length,
            ))
            if end >= duration:
                break
            index += 1

    out = pd.DataFrame(rows, columns=WINDOW_COLUMNS)
    if not out.empty:
        out = out.sort_values(
            ["centre", "subject", "session", "run", "start_s"], kind="mergesort"
        ).reset_index(drop=True)
    if save_per_recording:
        _save_windows_per_recording(out, per_recording_format, name="evaluation_windows")
    return out


def _nearest_seizure_distance(
    window_start: float,
    window_end: float,
    seizure_intervals: List[Tuple[float, float, Optional[str]]],
) -> float:
    """Return the shortest gap from a window to any seizure interval."""
    if not seizure_intervals:
        return float("inf")

    distances = []
    for seizure_start, seizure_end, _ in seizure_intervals:
        if window_end <= seizure_start:
            distances.append(seizure_start - window_end)
        elif window_start >= seizure_end:
            distances.append(window_start - seizure_end)
        else:
            distances.append(0.0)
    return float(max(0.0, min(distances)))


def _nearest_recording_edge_distance(
    window_start: float,
    window_end: float,
    recording_start: float,
    recording_end: float,
) -> float:
    """Return the shortest gap from a window to the recording start/end."""
    return float(max(0.0, min(window_start - recording_start, recording_end - window_end)))
