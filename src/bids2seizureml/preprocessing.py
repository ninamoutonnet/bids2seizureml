import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union
import h5py

os.environ.setdefault("MNE_USE_NUMBA", "false")

import mne
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingParams:
    dataset_path: Union[str, Path]
    montage: Optional[str] = None
    channels: Optional[List[str]] = None
    notch_filter: Optional[float] = None
    bandpass_filter: Optional[Tuple[float, float]] = None
    target_sampling_freq: Optional[float] = None
    debug: bool = False
    debug_limit: int = 10
    parallel: bool = False
    n_jobs: Optional[int] = None
    output_format: str = "edf"


class PreprocessingPipeline:
    """Preprocessing pipeline for BIDS EEG datasets using MNE.

    For each EDF file, applies optional channel selection, resampling, notch
    and bandpass filtering, then saves a processed recording under
    derivatives/preprocessing while mirroring the original structure. The
    output_format accepts "edf" (the default) or "h5".
    """

    def __init__(self, params: PreprocessingParams):
        self.params = params
        self.dataset_path = Path(params.dataset_path).expanduser().resolve()
        # Derivatives folder named after preprocessing parameters
        self.output_dir = self._build_output_dir()

    def _normalized_params_for_signature(self) -> dict:
        p = asdict(self.params)
        # The signature identifies *what preprocessing was done*, not which
        # dataset it ran on. Dataset identity is already encoded by location
        # (the derivatives folder lives inside the dataset), so including the
        # path here only makes the signature fragile to moves/renames without
        # adding any disambiguation. Drop it.
        p.pop("dataset_path", None)
        # Normalize montage token
        montage_norm = (p.get("montage") or "").strip()
        p["montage"] = montage_norm
        # Normalize channels to canonical uppercase form and make order-insensitive by sorting
        if p.get("channels"):
            normalized = [self._normalize_label(ch) for ch in p["channels"]]
            p["channels"] = sorted(normalized)
        # Remove debug flags from signature
        p.pop("debug", None)
        p.pop("debug_limit", None)
        # Remove execution strategy from signature so parallelism doesn't affect identity
        p.pop("parallel", None)
        p.pop("n_jobs", None)
        p["output_format"] = self._normalise_output_format(p.get("output_format", "edf"))
        return p

    def _build_output_dir(self) -> Path:
        base = self.dataset_path / "derivatives"
        sig_params = self._normalized_params_for_signature()
        sig = compute_signature_from_params_dict(sig_params)
        dir_name = f"preprocessing_{sig}"
        return base / dir_name

    @staticmethod
    def _normalise_output_format(output_format: object) -> str:
        fmt = str(output_format).strip().lower()
        if fmt not in {"edf", "h5"}:
            raise ValueError("output_format must be 'edf' or 'h5'")
        return fmt

    @staticmethod
    def _normalize_label(s: str) -> str:
        """Return a canonical channel label for comparison and renaming.

        - Trims whitespace, uppercases
        - Normalizes unicode dashes to ASCII '-'
        - Remaps temporal synonyms: T7->T3, T8->T4, P7->T5, P8->T6
        - Applies remapping token-wise to bipolar labels (e.g., F7-T7 -> F7-T3)
        """
        text = str(s).strip().upper()
        text = text.replace("\u2013", "-").replace("\u2212", "-").replace("\u2014", "-")
        mapping = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
        if "-" in text:
            a, b = [p.strip() for p in text.split("-", 1)]
            return f"{mapping.get(a, a)}-{mapping.get(b, b)}"
        return mapping.get(text, text)

    @staticmethod
    def _match_channels_upper(available: List[str], requested: List[str]) -> Tuple[List[str], List[str]]:
        """Upper-case exact matching with dash normalization."""
        upper_to_orig = {PreprocessingPipeline._normalize_label(name): name for name in available}
        matched: List[str] = []
        missing: List[str] = []
        for req in requested:
            orig = upper_to_orig.get(PreprocessingPipeline._normalize_label(req))
            if orig is not None:
                matched.append(orig)
            else:
                missing.append(req)
        return matched, missing

    def _rename_temporal_synonyms(self, raw: mne.io.BaseRaw) -> None:
        """Rename channels in-place to canonical temporal labels.
        Skips any rename that would cause a duplicate channel name, with a warning.
        """
        current_names = list(raw.ch_names)
        target_names = [self._normalize_label(n) for n in current_names]
        # Detect collisions
        existing = set(current_names)
        rename_map = {}
        for old, new in zip(current_names, target_names):
            if new == old:
                continue
            if new in existing and new not in rename_map.values():
                # Would collide with an existing channel; skip
                self._warn(f"skipping rename '{old}' -> '{new}' to avoid duplicate name")
                continue
            rename_map[old] = new
            existing.add(new)
        if rename_map:
            raw.rename_channels(rename_map)
            self._log(f"Renamed channels (temporal synonyms): {rename_map}")

    # Standard 10-20 (+10-10) scalp electrodes, in canonical form. Used to tell a
    # true bipolar pair (e.g. "F7-T3") apart from a referential label whose suffix
    # merely contains a dash (e.g. "EKG-REF", "PHOTIC-LE").
    _SCALP_ELECTRODES = frozenset({
        "FP1", "FP2", "FPZ", "F7", "F8", "F3", "F4", "FZ",
        "T3", "T4", "T5", "T6", "C3", "C4", "CZ",
        "P3", "P4", "PZ", "O1", "O2", "OZ", "A1", "A2",
    })

    @classmethod
    def _is_bipolar_pair(cls, name: str) -> bool:
        """True only if a hyphenated label joins two real scalp electrodes."""
        norm = cls._normalize_label(name)
        if "-" not in norm:
            return False
        a, b = norm.split("-", 1)
        return a in cls._SCALP_ELECTRODES and b in cls._SCALP_ELECTRODES

    _LONGITUDINAL_PAIRS = (
        ("FP1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
        ("FP2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
        ("FZ", "CZ"), ("CZ", "PZ"), ("FP1", "F3"), ("F3", "C3"),
        ("C3", "P3"), ("P3", "O1"), ("FP2", "F4"), ("F4", "C4"),
        ("C4", "P4"), ("P4", "O2"),
    )
    _MONTAGE_LABELS = {
        "longitudinal": "Longitudinal (double banana)",
        "cz": "Cz-referential",
        "average": "Common Average",
        "unknown": "Unknown",
    }

    @staticmethod
    def _classify_montage_target(target: str) -> str:
        """Interpret both source descriptions and requested montage aliases."""
        text = " ".join(str(target or "").lower().replace("_", " ").replace("-", " ").split())
        if text == "bipolar" or text.startswith(("longitudinal", "double banana")):
            return "longitudinal"
        if text == "cz" or text.startswith("cz "):
            return "cz"
        if text in {"avg", "car"} or text.startswith(("common average", "average")):
            return "average"
        return "unknown"

    def _infer_montage(self, ch_names: List[str], sidecar_meta: dict) -> str:
        """Use explicit metadata or consistent channel suffixes; never guess average."""
        ref = sidecar_meta.get("EEGReference") if isinstance(sidecar_meta, dict) else None
        if isinstance(ref, str):
            kind = self._classify_montage_target(ref)
            if kind != "unknown":
                return self._MONTAGE_LABELS[kind]
            if ref.strip().lower() not in {"", "n/a", "na", "none", "unknown"}:
                return "Unknown"

        names = [self._normalize_label(name) for name in ch_names]
        scalp = [name for name in names if name.split("-", 1)[0] in self._SCALP_ELECTRODES]
        if scalp and all(name.endswith("-AVG") for name in scalp):
            return self._MONTAGE_LABELS["average"]
        if scalp and all(name.endswith("-CZ") for name in scalp):
            return self._MONTAGE_LABELS["cz"]
        longitudinal = {f"{anode}-{cathode}" for anode, cathode in self._LONGITUDINAL_PAIRS}
        if scalp and all(name in longitudinal for name in scalp):
            return self._MONTAGE_LABELS["longitudinal"]
        return "Unknown"

    def _apply_montage(self, raw: mne.io.BaseRaw, target: str) -> mne.io.BaseRaw:
        """Convert average EEG to Cz or the complete longitudinal montage on a copy."""
        electrodes = {}
        for name in raw.ch_names:
            normalized = self._normalize_label(name)
            base = normalized[:-4] if normalized.endswith("-AVG") else normalized
            if base in self._SCALP_ELECTRODES:
                if base in electrodes:
                    raise ValueError(f"ambiguous duplicate electrode {base}")
                electrodes[base] = name

        if target == "longitudinal":
            pairs = self._LONGITUDINAL_PAIRS
        else:  # Cz uses every available scalp electrode except the reference itself.
            if "CZ" not in electrodes:
                raise ValueError("CZ or CZ-AVG is required for Cz referencing")
            pairs = [(base, "CZ") for base in electrodes if base != "CZ"]
        if not pairs:
            raise ValueError("no EEG channels available for referencing")
        required = {electrode for pair in pairs for electrode in pair}
        missing = sorted(required - electrodes.keys())
        if missing:
            raise ValueError(f"missing required electrodes: {', '.join(missing)}")

        names = [f"{anode}-{cathode}" for anode, cathode in pairs]
        converted = mne.set_bipolar_reference(
            raw,
            anode=[electrodes[anode] for anode, _ in pairs],
            cathode=[electrodes[cathode] for _, cathode in pairs],
            ch_name=names,
            drop_refs=True,
            copy=True,
            verbose="ERROR",
        )
        converted.pick(names)
        return converted

    def _resolve_montage(
        self,
        raw: mne.io.BaseRaw,
        reref_target: str,
        sidecar_meta: dict,
    ) -> Tuple[mne.io.BaseRaw, Optional[str], bool]:
        """Return the recording, confirmed montage label, and whether to skip.

        Matching montages pass through. Only common-average to Cz/longitudinal
        conversion is supported; unknown sources and failed conversions are
        skipped without changing the input recording.
        """
        if not reref_target:
            return raw, None, False
        want = self._classify_montage_target(reref_target)
        if want == "unknown":
            raise ValueError(f"Unsupported montage {reref_target!r}; use Longitudinal, Cz, or Common Average")
        source_label = self._infer_montage(list(raw.ch_names), sidecar_meta)
        current = self._classify_montage_target(source_label)
        if self.params.debug:
            self._log(f"Montage: current='{source_label}', target='{reref_target}'")
        if current == "unknown":
            self._warn("cannot determine the source reference; requested montage cannot be confirmed. Skipping file.")
            return raw, None, True

        if current == want:
            if want == "longitudinal":
                required = [f"{anode}-{cathode}" for anode, cathode in self._LONGITUDINAL_PAIRS]
                _, missing = self._match_channels_upper(raw.ch_names, required)
                if missing:
                    self._warn(f"incomplete longitudinal montage; missing channels: {', '.join(missing)}. Skipping file.")
                    return raw, None, True
            return raw, self._MONTAGE_LABELS[want], False

        if current != "average" or want not in {"cz", "longitudinal"}:
            self._warn(f"unsupported montage change from '{source_label}' to '{reref_target}'. Skipping file.")
            return raw, None, True
        try:
            converted = self._apply_montage(raw, want)
        except Exception as exc:
            self._warn(f"requested montage '{reref_target}' failed: {exc}. Skipping file.")
            return raw, None, True
        return converted, self._MONTAGE_LABELS[want], False


    def run(self) -> Path:
        # Checkpoint/resume behavior: process only missing outputs.
        # The derivatives folder is created lazily (at first successful write),
        # so a run that produces no outputs leaves no empty folder behind.
        self._configure_debug_logging()
        self._log("Starting preprocessing run (checkpoint mode: will skip already-processed files)")
        self._validate_parameters()
        edf_files = self._find_all_edf_files(self.dataset_path)
        if self.params.debug:
            edf_files = edf_files[: self.params.debug_limit]
            self._log(f"Debug enabled: limiting to {len(edf_files)} file(s)")

        processed: List[Path] = []
        # Precompute already processed files (for progress context)
        already_processed = 0
        for edf_path in edf_files:
            expected_out = self._construct_output_path(edf_path)
            if self._is_valid_output(expected_out):
                self._ensure_companion_files(edf_path, expected_out)
                already_processed += 1

        # Progress bar setup for non-debug mode
        use_progress = (not self.params.debug) and (tqdm is not None)
        pbar = None
        skipped_count = 0
        created_count = 0
        if use_progress:
            pbar = tqdm(total=len(edf_files), desc="Preprocessing EDFs", unit="file")
            pbar.set_postfix({
                "already": already_processed,
                "to_do": max(0, len(edf_files) - already_processed),
            })
        if getattr(self.params, "parallel", False):
            # Build list of files that still need processing
            to_process = []
            for edf_path in edf_files:
                expected_out = self._construct_output_path(edf_path)
                if self._is_valid_output(expected_out):
                    self._ensure_companion_files(edf_path, expected_out)
                    skipped_count += 1
                else:
                    to_process.append(edf_path)

            max_workers = self.params.n_jobs if (self.params.n_jobs and self.params.n_jobs > 0) else os.cpu_count() or 1
            self._log(f"Parallel mode enabled with {max_workers} worker(s); {len(to_process)} file(s) to process")

            # Submit tasks
            futures = []
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for edf_path in to_process:
                    futures.append(executor.submit(_process_single_file_worker, str(edf_path), asdict(self.params)))
                for fut in as_completed(futures):
                    try:
                        out_path_str, was_created = fut.result()
                        out_path = Path(out_path_str)
                        if self._is_valid_output(out_path):
                            processed.append(out_path)
                        if was_created:
                            created_count += 1
                    except Exception as exc:
                        logger.warning("worker failed with exception: %s", exc)
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({
                            "processed": created_count,
                            "skipped": skipped_count,
                            "total": len(edf_files),
                        })
        else:
            for edf_path in edf_files:
                # If this source already has an output, skip to support resume
                expected_out = self._construct_output_path(edf_path)
                if self._is_valid_output(expected_out):
                    self._ensure_companion_files(edf_path, expected_out)
                    self._log(f"Skipping (already processed): {expected_out}")
                    skipped_count += 1
                    if pbar is not None:
                        pbar.update(1)
                        pbar.set_postfix({
                            "processed": created_count,
                            "skipped": skipped_count,
                            "total": len(edf_files),
                        })
                    continue
                out_edf = self._process_single_file(edf_path)
                if self._is_valid_output(out_edf):
                    processed.append(out_edf)
                    created_count += 1
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({
                        "processed": created_count,
                        "skipped": skipped_count,
                        "total": len(edf_files),
                    })

        if pbar is not None:
            pbar.close()

        params_dict = asdict(self.params)
        if params_dict.get("dataset_path") is not None:
            params_dict["dataset_path"] = str(Path(params_dict["dataset_path"]).expanduser().resolve())
        # Normalize channel names in the log to the canonical uppercase form used in processing
        if params_dict.get("channels"):
            original_channels = list(params_dict["channels"])  # type: ignore[index]
            canonical_channels = [self._normalize_label(ch) for ch in original_channels]
            params_dict["channels"] = canonical_channels
        params_dict["output_format"] = self._normalise_output_format(
            params_dict.get("output_format", "edf")
        )

        # Count all outputs currently present (pre-existing + newly produced)
        total_outputs = 0
        for edf_path in edf_files:
            expected_out = self._construct_output_path(edf_path)
            if self._is_valid_output(expected_out):
                self._ensure_companion_files(edf_path, expected_out)
                total_outputs += 1

        run_log = {
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.output_dir),
            "num_input_files": len(edf_files),
            "num_output_files": int(total_outputs),
            "num_skipped_existing": int(skipped_count),
            "num_created_this_run": int(created_count),
            "parameters": params_dict,
        }
        if total_outputs == 0:
            # Nothing was producible (e.g. an incompatible montage conversion
            # for every file). Do not create the derivatives folder or write a
            # log — leaving an empty, log-bearing folder makes a failed run look
            # like a successful one and silently breaks downstream indexing.
            logger.warning(
                "No output files were created. The provided parameters may be "
                "incompatible with the data (dataset='%s', montage=%r). "
                "No derivatives folder was written.",
                self.dataset_path.name,
                self.params.montage,
            )
            self._log("Preprocessing finished with no outputs; derivatives folder not created.")
            return self.output_dir

        # At least one output exists -> ensure the root folder exists and write the log.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "PREPROCESSING_LOG.json").write_text(
            json.dumps(run_log, indent=2)
        )
        self._log(f"Wrote run log to {self.output_dir / 'PREPROCESSING_LOG.json'}")
        self._log("Preprocessing finished")
        return self.output_dir

    # --------------------------- helpers --------------------------- #
    def _validate_parameters(self) -> None:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {self.dataset_path}")
        self.params.output_format = self._normalise_output_format(self.params.output_format)
        if self.params.montage and self._classify_montage_target(self.params.montage) == "unknown":
            raise ValueError("montage must be Longitudinal, Cz, Common Average, or None")
        self._validate_frequencies()
        # NOTE: per-file channel availability is checked lazily in
        # _process_single_file (a file missing a requested channel is skipped
        # there). We deliberately do not pre-scan every EDF header here: it
        # doubles the header IO.
        self._log("Parameters validated")

    def _validate_frequencies(self, input_sfreq: Optional[float] = None) -> None:
        """Check requested frequencies against the rate used for filtering."""
        for name, value in (
            ("target_sampling_freq", self.params.target_sampling_freq),
            ("notch_filter", self.params.notch_filter),
        ):
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")

        if self.params.bandpass_filter is not None:
            low, high = self.params.bandpass_filter
            if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
                raise ValueError("bandpass_filter must be finite with 0 < low < high")

        sfreq = self.params.target_sampling_freq
        if sfreq is None:
            sfreq = input_sfreq
        if sfreq is None:
            return  # Without a target rate, check each recording after loading it.
        nyquist = sfreq / 2
        if self.params.notch_filter is not None and self.params.notch_filter >= nyquist:
            raise ValueError(f"notch_filter must be below {nyquist:g} Hz (half the sampling rate)")
        if self.params.bandpass_filter is not None and self.params.bandpass_filter[1] >= nyquist:
            raise ValueError(f"bandpass upper cutoff must be below {nyquist:g} Hz (half the sampling rate)")

    def _find_all_edf_files(self, root: Path) -> List[Path]:
        edf_files: List[Path] = []
        for p in root.rglob("*.edf"):
            parts = set(p.parts)
            # Skip anything under any derivatives directory, not only our output_dir
            if "derivatives" in parts:
                continue
            if p.stem.endswith("_desc-preprocessed"):
                continue
            edf_files.append(p)
        self._log(f"Found {len(edf_files)} EDF file(s) to process")
        return sorted(edf_files)

    def _construct_output_path(self, edf_path: Path) -> Path:
        try:
            relative = edf_path.relative_to(self.dataset_path)
        except ValueError:
            relative = Path(edf_path.name)
        out_dir = self.output_dir / relative.parent
        # NOTE: directory is intentionally NOT created here. This method only
        # computes the path (used by resume/exists checks). The directory is
        # created lazily at write time in _process_single_file, so that files
        # which are skipped (e.g. incompatible montage) leave no empty folders.
        suffix = ".h5" if self._normalise_output_format(self.params.output_format) == "h5" else ".edf"
        return out_dir / f"{edf_path.stem}_desc-preprocessed{suffix}"

    @staticmethod
    def _is_valid_output(path: Path) -> bool:
        """Return True only for completed output files.

        Some exporters create the destination path before raising, so a file
        existing on disk is not proof it holds data. Treating such a partial
        file as processed would make resume mode silently skip it. Both formats
        are therefore checked for actual content, not just non-zero size:

        - H5: the ``signals`` and ``channel_names`` datasets must exist and the
          signal array must be 2-D and non-empty.
        - EDF: the header must be readable and describe at least one channel and
          a non-zero recording duration (a header-only file that carries no data
          records is rejected). Only the header is read, not the sample data.
        """
        try:
            if not path.is_file() or path.stat().st_size <= 0:
                return False
            if path.suffix.lower() in {".h5", ".hdf5"}:
                with h5py.File(path, "r") as h5:
                    if "signals" not in h5 or "channel_names" not in h5:
                        return False
                    signals = h5["signals"]
                    return signals.ndim == 2 and signals.shape[0] > 0 and signals.shape[1] > 0
            # EDF: read header only (preload=False) and confirm it has channels
            # and a non-zero duration.
            raw_hdr = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
            return len(raw_hdr.ch_names) > 0 and raw_hdr.n_times > 0
        except OSError:
            return False
        except Exception:
            return False

    def _process_single_file(self, edf_path: Path) -> Path:
        self._log(f"Processing {edf_path}")
        out_edf = self._construct_output_path(edf_path)

        # Load EDF with MNE
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
        # First step: rename temporal electrode synonyms so downstream uses canonical labels
        self._rename_temporal_synonyms(raw)
        orig_sfreq = float(raw.info["sfreq"]) if raw.info.get("sfreq") else None
        orig_n_times = int(raw.n_times)
        self._log(f"Original sfreq={orig_sfreq}, samples={orig_n_times}")

        # Load original sidecar JSON if present for montage inference
        in_json = edf_path.with_suffix(".json")
        sidecar_meta = {}
        if in_json.exists():
            try:
                sidecar_meta = json.loads(in_json.read_text())
            except Exception:
                sidecar_meta = {}

        # Optional re-referencing (only from Common Average -> Cz or Longitudinal).
        # All montage decision logic lives in _resolve_montage; here we act on its
        # verdict: either skip this file, or continue with the applied label.
        reref_target = (self.params.montage or "").strip() if self.params.montage else ""
        raw, applied_reref, should_skip = self._resolve_montage(raw, reref_target, sidecar_meta)
        if should_skip:
            return out_edf

        # Channel selection: match once, then either skip (missing) or pick.
        if self.params.channels:
            requested_list = list(self.params.channels)
            if self.params.debug:
                req_upper = [str(ch).strip().upper() for ch in requested_list]
                avail_upper = [str(n).strip().upper() for n in raw.ch_names]
                self._log(f"Requested (upper): {req_upper}")
                self._log(f"Available (upper): {avail_upper}")
            matched, missing = self._match_channels_upper(raw.ch_names, requested_list)
            if missing:
                self._warn(f"requested channels missing in {edf_path}: {missing}. Skipping file.")
                # Do not write output for this file
                return out_edf
            if not matched:
                self._log("Requested channels not present; keeping original channels")
            else:
                raw.pick(matched)
                self._log(f"Selected channels: {matched}")

        if not self._apply_signal_processing(raw, edf_path):
            return out_edf

        # Export processed data
        metadata = self._build_output_metadata(raw, sidecar_meta, applied_reref)
        source_events = self._source_events_path(edf_path)
        self._write_processed_recording(raw, out_edf, metadata, edf_path, source_events)
        self._log(f"Wrote processed recording to {out_edf}")

        self._ensure_companion_files(edf_path, out_edf)

        return out_edf

    def _apply_signal_processing(self, raw: mne.io.BaseRaw, edf_path: Path) -> bool:
        """Apply all requested operations; a failure prevents this file being saved.

        Serial runs and parallel workers use this same failure policy. Changes
        made in memory before a failure are discarded with the recording.
        """
        operation = "frequency validation"
        try:
            self._validate_frequencies(raw.info["sfreq"])
            target = self.params.target_sampling_freq
            if target is not None and abs(raw.info["sfreq"] - target) > 1e-6:
                operation = "resampling"
                self._log(f"Resampling from {raw.info['sfreq']} Hz to {target} Hz")
                raw.resample(target)

            if self.params.notch_filter is not None:
                operation = "notch filter"
                self._log(f"Applying notch filter at {self.params.notch_filter} Hz")
                raw.notch_filter(freqs=[float(self.params.notch_filter)], verbose="ERROR")

            if self.params.bandpass_filter is not None:
                operation = "bandpass filter"
                low, high = self.params.bandpass_filter
                self._log(f"Applying bandpass filter {low}-{high} Hz")
                raw.filter(l_freq=float(low), h_freq=float(high), verbose="ERROR")
        except Exception as exc:
            self._warn(f"{edf_path}: {operation} failed: {exc}. Skipping file without saving.")
            return False
        return True

    def _build_output_metadata(
        self,
        raw: mne.io.BaseRaw,
        sidecar_meta: dict,
        applied_reref: Optional[str],
    ) -> dict:
        metadata = dict(sidecar_meta)

        metadata["SamplingFrequency"] = float(raw.info["sfreq"])
        metadata["RecordingDuration"] = float(raw.n_times / raw.info["sfreq"])
        if applied_reref:
            kind = self._classify_montage_target(applied_reref)
            metadata["EEGReference"] = self._MONTAGE_LABELS[kind]

        # A notch setting does not change the power-line frequency at acquisition.
        metadata.setdefault("PowerLineFrequency", "n/a")

        software_filters = {}
        if self.params.bandpass_filter:
            low, high = self.params.bandpass_filter
            software_filters["high-pass filter"] = {"half-amplitude cutoff (Hz)": float(low)}
            software_filters["low-pass filter"] = {"half-amplitude cutoff (Hz)": float(high)}
        if self.params.notch_filter:
            software_filters["notch filter"] = {
                "half-amplitude cutoff (Hz)": float(self.params.notch_filter)
            }
        if software_filters:
            inherited = metadata.get("SoftwareFilters", "n/a")
            if isinstance(inherited, dict):
                combined = dict(inherited)
            elif inherited not in (None, "", "n/a"):
                combined = {"inherited filter description": inherited}
            else:
                combined = {}
            for name, details in software_filters.items():
                # Preserve earlier applications of the same filter type.
                key = name
                number = 1
                while key in combined:
                    key = f"{name} (preprocessing {number})"
                    number += 1
                combined[key] = details
            metadata["SoftwareFilters"] = combined
        else:
            metadata["SoftwareFilters"] = metadata.get("SoftwareFilters", "n/a")

        # "Channels" is authoritative: it is taken directly from the processed
        # recording, so it already reflects any renaming, re-referencing and
        # channel selection, in canonical form. Drop legacy channel-name keys so
        # the sidecar has a single, unambiguous channel list.
        final_channels = list(raw.ch_names)
        metadata["EEGChannelCount"] = len(final_channels)
        metadata["Channels"] = final_channels
        for key in ("EEGChannels", "ChannelNames"):
            metadata.pop(key, None)

        return metadata

    def _write_processed_recording(
        self,
        raw: mne.io.BaseRaw,
        out_path: Path,
        metadata: dict,
        source_edf: Path,
        source_events: Path,
    ) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self._normalise_output_format(self.params.output_format) == "h5":
            self._write_h5(raw, out_path, metadata, source_edf, source_events)
        else:
            self._write_edf(raw, out_path, metadata)

    def _write_edf(self, raw: mne.io.BaseRaw, out_edf: Path, metadata: dict) -> None:
        tmp_edf = out_edf.with_name(f"{out_edf.stem}.tmp.edf")
        for stale_path in (tmp_edf, out_edf):
            if stale_path.exists() and stale_path.stat().st_size == 0:
                stale_path.unlink()
        try:
            mne.export.export_raw(str(tmp_edf), raw, fmt="edf", overwrite=True)
            if not self._is_valid_output(tmp_edf):
                raise IOError(f"EDF export produced no data: {tmp_edf}")
            tmp_edf.replace(out_edf)
        except Exception:
            if tmp_edf.exists() and tmp_edf.stat().st_size == 0:
                tmp_edf.unlink()
            if out_edf.exists() and out_edf.stat().st_size == 0:
                out_edf.unlink()
            raise

        out_json = out_edf.with_suffix(".json")
        out_json.write_text(json.dumps(metadata, indent=2))
        self._log(f"Wrote updated JSON sidecar to {out_json}")

    def _write_h5(
        self,
        raw: mne.io.BaseRaw,
        out_h5: Path,
        metadata: dict,
        source_edf: Path,
        source_events: Path,
    ) -> None:
        tmp_h5 = out_h5.with_name(f"{out_h5.stem}.tmp.h5")
        for stale_path in (tmp_h5, out_h5):
            if stale_path.exists() and stale_path.stat().st_size == 0:
                stale_path.unlink()

        data = np.ascontiguousarray(raw.get_data())
        sfreq = float(raw.info["sfreq"])
        n_channels, n_samples = data.shape
        samples_per_chunk = min(n_samples, max(1, int(round(sfreq * 60))))
        chunks = (n_channels, samples_per_chunk)
        string_dtype = h5py.string_dtype(encoding="utf-8")

        try:
            with h5py.File(tmp_h5, "w") as h5:
                h5.attrs["schema"] = "preprocessing_pipeline.h5.v1"
                h5.attrs["signal_layout"] = "channels_by_samples"
                h5.attrs["sampling_frequency"] = sfreq
                h5.attrs["n_channels"] = int(n_channels)
                h5.attrs["n_samples"] = int(n_samples)
                h5.attrs["duration_s"] = float(n_samples / sfreq)
                h5.attrs["data_unit"] = "V"
                h5.attrs["source_edf_path"] = str(source_edf)
                h5.attrs["source_events_path"] = str(source_events) if source_events.exists() else ""

                h5.create_dataset("signals", data=data, chunks=chunks)
                h5.create_dataset(
                    "channel_names",
                    data=np.asarray(list(raw.ch_names), dtype=object),
                    dtype=string_dtype,
                )
                h5.create_dataset(
                    "metadata_json",
                    data=json.dumps(self._json_safe(metadata), sort_keys=True),
                    dtype=string_dtype,
                )
                h5.create_dataset(
                    "preprocessing_params_json",
                    data=json.dumps(self._params_for_metadata(), sort_keys=True),
                    dtype=string_dtype,
                )
                if source_events.exists():
                    h5.create_dataset(
                        "events_tsv",
                        data=source_events.read_text(),
                        dtype=string_dtype,
                    )

            if not self._is_valid_output(tmp_h5):
                raise IOError(f"H5 export produced no data: {tmp_h5}")
            tmp_h5.replace(out_h5)
        except Exception:
            if tmp_h5.exists() and tmp_h5.stat().st_size == 0:
                tmp_h5.unlink()
            if out_h5.exists() and out_h5.stat().st_size == 0:
                out_h5.unlink()
            raise

        out_json = out_h5.with_suffix(".json")
        out_json.write_text(json.dumps(metadata, indent=2))
        self._log(f"Wrote updated JSON sidecar to {out_json}")

    def _ensure_companion_files(self, source_edf: Path, out_path: Path) -> None:
        """Restore missing HDF5 JSON and events files without processing signals."""
        self._ensure_json_sidecar(out_path)
        source_events = self._source_events_path(source_edf)
        dest_events = out_path.parent / source_events.name
        if dest_events.exists():
            return
        tmp_events = dest_events.with_name(f"{dest_events.name}.tmp")
        try:
            if source_events.exists():
                shutil.copy2(source_events, tmp_events)
            elif out_path.suffix.lower() in {".h5", ".hdf5"}:
                with h5py.File(out_path, "r") as h5:
                    if "events_tsv" not in h5:
                        return
                    events_text = h5["events_tsv"][()]
                if isinstance(events_text, bytes):
                    events_text = events_text.decode("utf-8")
                tmp_events.write_text(events_text)
            else:
                return  # The source recording has no events file to copy.
            # An interrupted copy leaves a temporary file that resume can retry.
            tmp_events.replace(dest_events)
            self._log(f"Restored events to {dest_events}")
        except Exception as exc:
            self._warn(f"could not restore events for {out_path}: {exc}")

    def _ensure_json_sidecar(self, out_path: Path) -> None:
        if out_path.suffix.lower() not in {".h5", ".hdf5"}:
            return
        out_json = out_path.with_suffix(".json")
        if out_json.exists():
            return
        try:
            with h5py.File(out_path, "r") as h5:
                if "metadata_json" not in h5:
                    return
                raw_json = h5["metadata_json"][()]
                if isinstance(raw_json, bytes):
                    raw_json = raw_json.decode("utf-8")
                metadata = json.loads(str(raw_json))
            out_json.write_text(json.dumps(metadata, indent=2))
            self._log(f"Backfilled JSON sidecar for existing H5: {out_json}")
        except Exception as exc:
            self._warn(f"could not backfill JSON sidecar for {out_path}: {exc}")

    def _params_for_metadata(self) -> dict:
        params_dict = asdict(self.params)
        if params_dict.get("dataset_path") is not None:
            params_dict["dataset_path"] = str(
                Path(params_dict["dataset_path"]).expanduser().resolve()
            )
        if params_dict.get("channels"):
            params_dict["channels"] = [
                self._normalize_label(ch) for ch in list(params_dict["channels"])
            ]
        params_dict["output_format"] = self._normalise_output_format(
            params_dict.get("output_format", "edf")
        )
        return self._json_safe(params_dict)

    @staticmethod
    def _json_safe(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): PreprocessingPipeline._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [PreprocessingPipeline._json_safe(v) for v in value]
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return value

    @staticmethod
    def _source_events_path(edf_path: Path) -> Path:
        stem = edf_path.stem
        base = stem[:-4] if stem.endswith("_eeg") else stem
        return edf_path.with_name(f"{base}_events.tsv")

  
    _DEBUG_HANDLER_NAME = "preprocessing_debug_handler"
    def _configure_debug_logging(self) -> None:
        """Make this module's logging reflect ``params.debug`` on every run.

        Libraries should not configure logging for the whole application, so by
        default the pipeline emits through ``logger`` and stays silent unless the
        caller has set up handlers. As a convenience, ``debug=True`` lowers this
        module's logger to DEBUG and attaches a handler *only if the application
        has not configured logging itself* (no handlers on the root logger).

        The state is symmetric: ``debug=False`` removes the handler the pipeline
        previously added and resets the logger level, so a debug run does not
        leak DEBUG output into later non-debug runs in the same process (e.g. the
        same notebook kernel). Handlers the application attached are never touched.
        """
        # Always remove any debug handler this pipeline added on a prior run.
        for h in list(logger.handlers):
            if getattr(h, "name", None) == self._DEBUG_HANDLER_NAME:
                logger.removeHandler(h)

        if not self.params.debug:
            # Reset our own level override; NOTSET lets the logger inherit again.
            logger.setLevel(logging.NOTSET)
            return

        logger.setLevel(logging.DEBUG)
        root_has_handlers = bool(logging.getLogger().handlers)
        if not root_has_handlers and not logger.handlers:
            handler = logging.StreamHandler()
            handler.name = self._DEBUG_HANDLER_NAME
            handler.setFormatter(logging.Formatter("[%(name)s] %(levelname)s %(message)s"))
            logger.addHandler(handler)
            
    
    def _log(self, message: str) -> None:
        """Emit an informational message at DEBUG level.

        Routed through the standard ``logging`` module so the calling
        application controls verbosity and destination. When ``params.debug``
        is set, the pipeline lowers this logger's level in ``run()`` so these
        messages surface without the caller configuring logging.
        """
        logger.debug(message)

    def _warn(self, message: str) -> None:
        """Emit a warning (e.g. a skipped file or a failed optional step).

        Unlike ``_log``, warnings are emitted at WARNING level so they are
        visible to any application that has configured logging at all, not only
        in debug mode. A recording being skipped is something the caller should
        be able to see by default.
        """
        logger.warning(message)


def compute_signature_from_params_dict(params: dict, *, hex_len: int = 32) -> str:
    """Compute a deterministic signature (SHA-256) from normalized params.

    - Expects a normalized params dict as produced by _normalized_params_for_signature.
    - Uses json dumps with sort_keys=True to ensure stable ordering.
    - Returns the first `hex_len` hex chars of the SHA-256 digest (default 32).
    """
    sig_json = json.dumps(params, sort_keys=True)
    digest = hashlib.sha256(sig_json.encode("utf-8")).hexdigest()
    return digest[:hex_len]


def signature_for_params(
    *,
    dataset_path: Union[str, Path],
    montage: Optional[str] = None,
    channels: Optional[Iterable[str]] = None,
    notch_filter: Optional[float] = None,
    bandpass_filter: Optional[Tuple[float, float]] = None,
    target_sampling_freq: Optional[float] = None,
    output_format: str = "edf",
    hex_len: int = 32,
) -> str:
    """Public helper to compute the preprocessing signature from raw parameters.

    Normalizes and sorts channels order-insensitively, trims montage, and
    computes a SHA-256-based signature. The signature depends only on the
    preprocessing parameters, not on the dataset path (dataset identity comes
    from the derivatives folder's location).
    """
    params = PreprocessingParams(
        dataset_path=dataset_path,
        montage=montage,
        channels=list(channels) if channels is not None else None,
        notch_filter=notch_filter,
        bandpass_filter=bandpass_filter,
        target_sampling_freq=target_sampling_freq,
        output_format=output_format,
    )
    # Reuse pipeline's normalization logic for consistency
    pipeline = PreprocessingPipeline(params)
    normalized = pipeline._normalized_params_for_signature()
    return compute_signature_from_params_dict(normalized, hex_len=hex_len)


def derivatives_dir_for_params(
    *,
    dataset_path: Union[str, Path],
    montage: Optional[str] = None,
    channels: Optional[Iterable[str]] = None,
    notch_filter: Optional[float] = None,
    bandpass_filter: Optional[Tuple[float, float]] = None,
    target_sampling_freq: Optional[float] = None,
    output_format: str = "edf",
    hex_len: int = 32,
) -> Path:
    """Return the derivatives directory path for a given parameter set."""
    root = Path(dataset_path).expanduser().resolve()
    sig = signature_for_params(
        dataset_path=root,
        montage=montage,
        channels=channels,
        notch_filter=notch_filter,
        bandpass_filter=bandpass_filter,
        target_sampling_freq=target_sampling_freq,
        output_format=output_format,
        hex_len=hex_len,
    )
    return root / "derivatives" / f"preprocessing_{sig}"


def preprocess_bids(
    dataset_path: Union[str, Path],
    *,
    montage: Optional[str] = None,
    channels: Optional[Iterable[str]] = None,
    notch_filter: Optional[float] = None,
    bandpass_filter: Optional[Tuple[float, float]] = None,
    target_sampling_freq: Optional[float] = None,
    output_format: str = "edf",
    debug: bool = False,
    debug_limit: int = 10,
    parallel: bool = False,
    n_jobs: Optional[int] = None,
) -> Path:
    """Preprocess a BIDS EEG dataset with output_format="edf" or "h5".

    Example:
        out_dir = preprocess_bids("/path/to/bids", bandpass_filter=(0.5, 50), debug=True)
    """
    params = PreprocessingParams(
        dataset_path=dataset_path,
        montage=montage,
        channels=list(channels) if channels is not None else None,
        notch_filter=notch_filter,
        bandpass_filter=bandpass_filter,
        target_sampling_freq=target_sampling_freq,
        output_format=output_format,
        debug=debug,
        debug_limit=debug_limit,
        parallel=parallel,
        n_jobs=n_jobs,
    )
    pipeline = PreprocessingPipeline(params)
    return pipeline.run()


def _process_single_file_worker(edf_path_str: str, params_dict: dict) -> Tuple[str, bool]:
    """Top-level worker function to enable process-based parallelism.

    Returns a tuple of (output_path_str, was_created_bool).
    """
    # Reconstruct params and pipeline in the worker process
    params = PreprocessingParams(**params_dict)
    pipeline = PreprocessingPipeline(params)
    out_edf = pipeline._process_single_file(Path(edf_path_str))
    # "Created" if a valid output now exists.
    was_created = pipeline._is_valid_output(out_edf)
    return str(out_edf), bool(was_created)
