"""Read current fold tables as PyTorch datasets, with optional shared window caches.

EEG is read in volts from EDF/HDF5. Output is float32 (channels, samples),
with labels background=0, seizure=1, mixed=-100 (CrossEntropyLoss's ignore index).
Other losses/scoring must mask mixed labels explicitly. Fractions remain in metadata.
Partial windows are padded to window_length_s. Transforms use valid samples only.
"""

import atexit
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .transforms import (
    apply_augmentation_plan,
    build_augmentation_plan,
    normalise_eeg,
    original_row_index,
)


_CACHE_VERSION = 3
_READER_CACHE_SIZE = max(1, int(os.environ.get("EEG_READER_CACHE_SIZE", "8")))
_EDF_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_H5_CACHE: "OrderedDict[str, Any]" = OrderedDict()
_SPLITS = ("train", "val", "test")


class SignalWindowDataset(Dataset):
    """Lazy dataset over fold rows pointing to EDF or H5 signals."""

    def __init__(
        self,
        rows: Iterable[Any],
        *,
        normalisation: Optional[str] = None,
        augmentation: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
    ):
        self.rows = list(rows)
        self.normalisation = normalisation
        self.augmentation_plan = build_augmentation_plan(self.rows, augmentation)
        self.return_metadata = bool(return_metadata)

    def __len__(self) -> int:
        return len(self.augmentation_plan)

    def __getitem__(self, idx: int):
        plan_item = self.augmentation_plan[idx]
        row_idx = original_row_index(plan_item, idx)
        row = self.rows[row_idx]
        x, valid_samples = _read_window(row)
        x = _prepare_window(x, valid_samples, plan_item, self.normalisation)
        y = _label_to_int(getattr(row, "label", "background"))
        x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        y_t = torch.tensor(y, dtype=torch.long)
        if self.return_metadata:
            return x_t, y_t, _metadata(row, row_idx, idx, plan_item, valid_samples, x.shape[1])
        return x_t, y_t


class IndexedWindowDataset(Dataset):
    """Fold-specific dataset that indexes into a shared materialized store."""

    def __init__(
        self,
        rows: Sequence[Any],
        store_indices: Sequence[int],
        store: Any,
        *,
        normalisation: Optional[str] = None,
        augmentation: Optional[Dict[str, Any]] = None,
        return_metadata: bool = False,
    ):
        self.rows = list(rows)
        self.store_indices = [int(idx) for idx in store_indices]
        self.store = store
        self.normalisation = normalisation
        self.augmentation_plan = build_augmentation_plan(self.rows, augmentation)
        self.return_metadata = bool(return_metadata)

    def __len__(self) -> int:
        return len(self.augmentation_plan)

    def __getitem__(self, idx: int):
        plan_item = self.augmentation_plan[idx]
        row_idx = original_row_index(plan_item, idx)
        row = self.rows[row_idx]
        store_idx = self.store_indices[row_idx]
        x, y, valid_samples = self.store.get(store_idx)
        x = _prepare_window(x, valid_samples, plan_item, self.normalisation)
        x_t = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
        y_t = torch.tensor(int(y), dtype=torch.long)
        if self.return_metadata:
            meta = _metadata(row, row_idx, idx, plan_item, valid_samples, x.shape[1])
            meta["store_index"] = int(store_idx)
            return x_t, y_t, meta
        return x_t, y_t


class TensorWindowStore:
    """Shared window store backed by CPU tensors."""

    def __init__(self, x: torch.Tensor, y: torch.Tensor, valid_samples: torch.Tensor):
        self.x = x
        self.y = y
        self.valid_samples = valid_samples

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def get(self, idx: int) -> Tuple[np.ndarray, int, int]:
        x = self.x[int(idx)].detach().cpu().numpy().copy()
        y = int(self.y[int(idx)].reshape(()).item())
        return x, y, int(self.valid_samples[int(idx)].item())


class MemmapWindowStore:
    """Shared window store backed by `.npy` files opened lazily."""

    def __init__(self, x_path: Path, y_path: Path, valid_path: Path):
        self.x_path = str(x_path)
        self.y_path = str(y_path)
        self.valid_path = str(valid_path)
        self._x = None
        self._y = None
        self._valid = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_x"] = None
        state["_y"] = None
        state["_valid"] = None
        return state

    def __len__(self) -> int:
        return int(self._arrays()[1].shape[0])

    def get(self, idx: int) -> Tuple[np.ndarray, int, int]:
        x, y, valid = self._arrays()
        return np.array(x[int(idx)], copy=True), int(y[int(idx)]), int(valid[int(idx)])

    def _arrays(self):
        if self._x is None:
            self._x = np.load(self.x_path, mmap_mode="r")
        if self._y is None:
            self._y = np.load(self.y_path, mmap_mode="r")
        if self._valid is None:
            self._valid = np.load(self.valid_path, mmap_mode="r")
        return self._x, self._y, self._valid


def build_dataloaders_from_folds(
    folds: Dict[str, Dict[str, Any]],
    *,
    fold_name: Optional[str] = None,
    batch_size: int = 64,
    normalisation: Optional[str] = "global_zscore",
    augmentation: Optional[Dict[str, Any]] = None,
    materialize: Optional[str] = None,
    cache_dir: Optional[Any] = None,
    force_rebuild: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
    return_metadata: bool = False,
    shuffle_train: bool = True,
    drop_last_train: bool = False,
) -> Dict[str, Dict[str, Optional[DataLoader]]]:
    """Build train/val/test dataloaders for one fold or all folds.

    By default, normalise each window using one mean and standard deviation
    across all channels and valid samples. Pass normalisation=None to disable.

    materialize:
      None/'none' -> lazy file reading
      'memory'    -> preload each unique requested window into RAM once
      'disk'      -> cache each unique requested window as .pt tensors
      'memmap'    -> cache each unique requested window as .npy files

    Caches contain original windows, labels and valid-sample counts. Training
    augmentation and per-window normalisation happen on access, so changing
    these transforms does not require rewriting the cache. Disk mode loads its
    .pt cache into RAM; memmap accesses .npy slices without preloading the store.

    Use return_metadata=True for continuous evaluation. Mixed windows have
    y=-100 (the standard CrossEntropyLoss ignore index), and metadata preserves
    their label/fraction and valid_samples. No binary threshold or event scoring
    is imposed here. Model predictions can still be produced for every window.
    """

    folds = _coerce_folds(folds)
    selected = [fold_name] if fold_name is not None else list(folds.keys())
    datasets_by_fold = _build_datasets_for_selected_folds(
        folds,
        selected,
        normalisation=normalisation,
        augmentation=augmentation,
        materialize=materialize,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
        return_metadata=return_metadata,
    )

    out = {}
    for name in selected:
        out[name] = {
            split: _loader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train" and shuffle_train),
                drop_last=(split == "train" and drop_last_train),
                num_workers=num_workers,
                pin_memory=pin_memory,
                seed=_stable_seed(seed, name, split),
                return_metadata=return_metadata,
            )
            for split, dataset in datasets_by_fold[name].items()
        }
    return out


def build_datasets_from_folds(
    folds: Dict[str, Dict[str, Any]],
    *,
    fold_name: str,
    normalisation: Optional[str] = "global_zscore",
    augmentation: Optional[Dict[str, Any]] = None,
    materialize: Optional[str] = None,
    cache_dir: Optional[Any] = None,
    force_rebuild: bool = False,
    return_metadata: bool = False,
) -> Dict[str, Optional[Dataset]]:
    """Build train/val/test datasets for one fold, using global window z-score
    normalisation by default. Statistics use all channels and valid samples;
    pass normalisation=None to disable.
    """

    folds = _coerce_folds(folds)
    if fold_name not in folds:
        raise ValueError(f"Unknown fold_name={fold_name!r}. Available: {list(folds)}")
    return _build_datasets_for_selected_folds(
        folds,
        [fold_name],
        normalisation=normalisation,
        augmentation=augmentation,
        materialize=materialize,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
        return_metadata=return_metadata,
    )[fold_name]


def _build_datasets_for_selected_folds(
    folds,
    selected,
    *,
    normalisation,
    augmentation,
    materialize,
    cache_dir,
    force_rebuild,
    return_metadata,
):
    mode = _materialize_mode(materialize)
    if mode is None:
        return {
            name: _lazy_datasets_for_fold(
                folds,
                name,
                normalisation=normalisation,
                augmentation=augmentation,
                return_metadata=return_metadata,
            )
            for name in selected
        }
    return _shared_datasets_for_folds(
        folds,
        selected,
        normalisation=normalisation,
        augmentation=augmentation,
        materialize=mode,
        cache_dir=cache_dir,
        force_rebuild=force_rebuild,
        return_metadata=return_metadata,
    )


def _lazy_datasets_for_fold(folds, fold_name, *, normalisation, augmentation, return_metadata):
    if fold_name not in folds:
        raise ValueError(f"Unknown fold_name={fold_name!r}. Available: {list(folds)}")
    split = folds[fold_name]
    return {
        "train": _dataset_from_frame(
            split.get("train"),
            normalisation=normalisation,
            augmentation=augmentation,
            return_metadata=return_metadata,
        ),
        "val": _dataset_from_frame(
            split.get("val"),
            normalisation=normalisation,
            augmentation=None,
            return_metadata=return_metadata,
        ),
        "test": _dataset_from_frame(
            split.get("test"),
            normalisation=normalisation,
            augmentation=None,
            return_metadata=return_metadata,
        ),
    }


def _shared_datasets_for_folds(
    folds,
    selected,
    *,
    normalisation,
    augmentation,
    materialize,
    cache_dir,
    force_rebuild,
    return_metadata,
):
    split_rows = _rows_by_fold_and_split(folds, selected)
    unique_rows, index_by_key = _unique_rows(split_rows, selected)
    store = _materialize_window_store(
        unique_rows,
        mode=materialize,
        cache_root=_cache_root(cache_dir, unique_rows),
        force_rebuild=force_rebuild,
    )

    datasets = {}
    for fold_name in selected:
        datasets[fold_name] = {}
        for split in _SPLITS:
            rows = split_rows[fold_name][split]
            if not rows:
                datasets[fold_name][split] = None
                continue
            indices = [index_by_key[_window_key(row)] for row in rows]
            datasets[fold_name][split] = IndexedWindowDataset(
                rows,
                indices,
                store,
                normalisation=normalisation,
                augmentation=augmentation if split == "train" else None,
                return_metadata=return_metadata,
            )
    return datasets


def _dataset_from_frame(df, *, normalisation, augmentation, return_metadata):
    rows = _frame_rows(df)
    if not rows:
        return None
    return SignalWindowDataset(
        rows,
        normalisation=normalisation,
        augmentation=augmentation,
        return_metadata=return_metadata,
    )


def _coerce_folds(folds):
    if hasattr(folds, "folds"):
        folds = folds.folds
    if not hasattr(folds, "keys"):
        raise TypeError("folds must be a fold dictionary or a FoldDataFrames object with a .folds attribute")
    return folds


def _loader(dataset, *, batch_size, shuffle, drop_last, num_workers, pin_memory, seed, return_metadata):
    if dataset is None:
        return None
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=bool(num_workers > 0),
        collate_fn=_collate_with_metadata if return_metadata else None,
    )


def _collate_with_metadata(batch):
    xs, ys, metadata = zip(*batch)
    return torch.stack(list(xs), dim=0), torch.stack(list(ys), dim=0), list(metadata)


def _rows_by_fold_and_split(folds, selected):
    rows = {}
    for fold_name in selected:
        if fold_name not in folds:
            raise ValueError(f"Unknown fold_name={fold_name!r}. Available: {list(folds)}")
        rows[fold_name] = {
            split: _frame_rows(folds[fold_name].get(split))
            for split in _SPLITS
        }
    return rows


def _frame_rows(df) -> List[Any]:
    if df is None or len(df) == 0:
        return []
    # Plain namespaces also pickle correctly with spawned DataLoader workers.
    return [SimpleNamespace(**row._asdict()) for row in df.itertuples(index=False)]


def _unique_rows(split_rows, selected):
    unique = []
    index_by_key = {}
    for fold_name in selected:
        for split in _SPLITS:
            for row in split_rows[fold_name][split]:
                key = _window_key(row)
                if key not in index_by_key:
                    index_by_key[key] = len(unique)
                    unique.append(row)
    return unique, index_by_key


def _materialize_window_store(rows, *, mode, cache_root: Path, force_rebuild: bool):
    if not rows:
        return TensorWindowStore(torch.empty((0,), dtype=torch.float32),
                                 torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long))
    if mode == "memory":
        return TensorWindowStore(*_read_rows_to_tensors(rows))

    payload = _store_cache_payload(rows, mode)
    signature = _hash_json(payload)
    store_dir = cache_root / _safe_name("shared_windows", mode, signature[:16])
    manifest_path = store_dir / "manifest.json"
    if not force_rebuild and _manifest_valid(manifest_path, mode, signature):
        return _load_window_store(store_dir, mode)

    store_dir.mkdir(parents=True, exist_ok=True)
    # An interrupted rebuild must not leave an older manifest validating partial files.
    if manifest_path.exists():
        manifest_path.unlink()
    if mode == "disk":
        return _write_disk_store(rows, store_dir, payload, signature)
    if mode == "memmap":
        return _write_memmap_store(rows, store_dir, payload, signature)
    raise ValueError(f"Unknown materialize mode {mode!r}")


def _read_rows_to_tensors(rows):
    xs, ys, valid = [], [], []
    for row in rows:
        x, valid_samples = _read_window(row)
        xs.append(torch.from_numpy(x))
        ys.append(torch.tensor(_label_to_int(getattr(row, "label", "background")), dtype=torch.long))
        valid.append(valid_samples)
    return torch.stack(xs), torch.stack(ys), torch.tensor(valid, dtype=torch.long)


def _write_disk_store(rows, store_dir, payload, signature):
    x, y, valid = _read_rows_to_tensors(rows)
    torch.save(x, store_dir / "x.pt")
    torch.save(y, store_dir / "y.pt")
    torch.save(valid, store_dir / "valid_samples.pt")
    _write_json(store_dir / "windows.json", [_window_record(row) for row in rows])
    files = {
        "x": str(store_dir / "x.pt"),
        "y": str(store_dir / "y.pt"),
        "valid": str(store_dir / "valid_samples.pt"),
        "windows": str(store_dir / "windows.json"),
    }
    _write_manifest(store_dir, "disk", signature, payload, files)
    return TensorWindowStore(x, y, valid)


def _write_memmap_store(rows, store_dir, payload, signature):
    first_x, first_valid = _read_window(rows[0])
    first_y = np.asarray(_label_to_int(getattr(rows[0], "label", "background")), dtype=np.int64)
    x_path = store_dir / "x.npy"
    y_path = store_dir / "y.npy"
    valid_path = store_dir / "valid_samples.npy"
    x_mm = np.lib.format.open_memmap(x_path, mode="w+", dtype=first_x.dtype, shape=(len(rows), *first_x.shape))
    y_mm = np.lib.format.open_memmap(y_path, mode="w+", dtype=first_y.dtype, shape=(len(rows),))
    valid_mm = np.lib.format.open_memmap(valid_path, mode="w+", dtype=np.int64, shape=(len(rows),))
    x_mm[0] = first_x
    y_mm[0] = first_y
    valid_mm[0] = first_valid
    for idx, row in enumerate(rows[1:], start=1):
        x, valid_samples = _read_window(row)
        if x.shape != first_x.shape:
            raise ValueError("Shared caches require matching channel counts and window sample lengths")
        x_mm[idx] = x
        y_mm[idx] = np.asarray(_label_to_int(getattr(row, "label", "background")), dtype=np.int64)
        valid_mm[idx] = valid_samples
    x_mm.flush()
    y_mm.flush()
    valid_mm.flush()
    del x_mm, y_mm, valid_mm

    _write_json(store_dir / "windows.json", [_window_record(row) for row in rows])
    files = {
        "x": str(x_path),
        "y": str(y_path),
        "valid": str(valid_path),
        "windows": str(store_dir / "windows.json"),
    }
    _write_manifest(store_dir, "memmap", signature, payload, files)
    return MemmapWindowStore(x_path, y_path, valid_path)


def _load_window_store(store_dir: Path, mode: str):
    manifest = json.loads((store_dir / "manifest.json").read_text())
    files = manifest["files"]
    if mode == "memmap":
        return MemmapWindowStore(Path(files["x"]), Path(files["y"]), Path(files["valid"]))
    try:
        x = torch.load(files["x"], map_location="cpu", weights_only=True)
        y = torch.load(files["y"], map_location="cpu", weights_only=True)
        valid = torch.load(files["valid"], map_location="cpu", weights_only=True)
    except TypeError:
        x = torch.load(files["x"], map_location="cpu")
        y = torch.load(files["y"], map_location="cpu")
        valid = torch.load(files["valid"], map_location="cpu")
    return TensorWindowStore(x, y, valid)


def _read_window(row: Any) -> Tuple[np.ndarray, int]:
    """Read valid recording time, then pad to the model's nominal input duration."""
    path = _signal_path(row)
    fmt = _signal_format(row, path)
    start_s = float(getattr(row, "start_s"))
    end_s = float(getattr(row, "end_s"))
    window_length = float(getattr(row, "window_length_s", end_s - start_s))
    valid_duration = float(getattr(row, "valid_duration_s", end_s - start_s))
    if not (np.isfinite(start_s) and np.isfinite(end_s) and 0 <= start_s < end_s
            and np.isfinite(window_length) and np.isfinite(valid_duration)
            and 0 < valid_duration <= window_length + 1e-6):
        raise ValueError("Window times must be finite with 0 <= start < end and 0 < valid duration <= window length")
    end_s = min(end_s, start_s + min(valid_duration, window_length))
    if fmt == "h5":
        data, sfreq = _read_h5_window(path, start_s, end_s)
    elif fmt == "edf":
        data, sfreq = _read_edf_window(path, start_s, end_s)
    else:
        raise ValueError(f"Unsupported signal format {fmt!r} for {path}")
    valid_samples = data.shape[1]
    # Ceil keeps every valid sample when the requested duration is fractional in samples.
    length = int(np.ceil(window_length * sfreq))
    if not 0 < valid_samples <= length:
        raise ValueError(f"Invalid sample count for window in {path}: {valid_samples} valid, {length} requested")
    return _pad_to_length(data, data.shape[0], length), valid_samples


def _read_h5_window(path: str, start_s: float, end_s: float) -> Tuple[np.ndarray, float]:
    h5 = _cached_h5(path)
    signals = h5["signals"]
    sfreq = float(h5.attrs["sampling_frequency"])
    start, stop = _sample_bounds(start_s, end_s, sfreq)
    stop_available = min(stop, signals.shape[1])
    if start < signals.shape[1] and stop_available > start:
        data = np.asarray(signals[:, start:stop_available], dtype=np.float32)
    else:
        data = np.zeros((signals.shape[0], 0), dtype=np.float32)
    return data, sfreq


def _read_edf_window(path: str, start_s: float, end_s: float) -> Tuple[np.ndarray, float]:
    reader = _cached_edf(path)
    sfreq = float(reader.info["sfreq"])
    start, stop = _sample_bounds(start_s, end_s, sfreq)
    stop = min(stop, reader.n_times)
    if start >= stop:
        return np.zeros((len(reader.ch_names), 0), dtype=np.float32), sfreq
    # MNE returns volts, matching the HDF5 signal contract (EDF may store µV).
    return np.asarray(reader.get_data(start=start, stop=stop), dtype=np.float32), sfreq


def _prepare_window(x, valid_samples, plan_item, normalisation):
    """Transform valid samples only and keep the padded tail exactly zero."""
    valid = apply_augmentation_plan(x[:, :valid_samples], plan_item)
    valid = normalise_eeg(valid, normalisation)
    return _pad_to_length(valid, x.shape[0], x.shape[1])


def _sample_bounds(start_s: float, end_s: float, sfreq: float):
    # Round to nearest sample, with exact half-samples consistently rounded up.
    start = int(np.floor(float(start_s) * sfreq + 0.5))
    stop = int(np.floor(float(end_s) * sfreq + 0.5))
    return max(start, 0), max(stop, start)


def _pad_to_length(data: np.ndarray, n_channels: int, length: int) -> np.ndarray:
    if data.shape == (n_channels, length):
        return np.ascontiguousarray(data, dtype=np.float32)
    out = np.zeros((n_channels, length), dtype=np.float32)
    out[:, : min(length, data.shape[1])] = data[:, : min(length, data.shape[1])]
    return out


def _signal_path(row: Any) -> str:
    for field in ("signal_path", "edf_path"):
        value = getattr(row, field, None)
        if value and str(value) != "nan":
            return str(value)
    raise FileNotFoundError("Fold row has no signal_path or edf_path")


def _signal_format(row: Any, path: str) -> str:
    value = getattr(row, "signal_format", None)
    if value and str(value) != "nan":
        return "h5" if str(value).lower() in {"h5", "hdf5"} else str(value).lower()
    suffix = Path(path).suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        return "h5"
    if suffix == ".edf":
        return "edf"
    return suffix.lstrip(".")


def _cached_h5(path: str):
    h5 = _H5_CACHE.get(path)
    if h5 is None:
        h5 = h5py.File(path, "r")
        _H5_CACHE[path] = h5
    _H5_CACHE.move_to_end(path)
    while len(_H5_CACHE) > _READER_CACHE_SIZE:
        _, old = _H5_CACHE.popitem(last=False)
        old.close()
    return h5


def _cached_edf(path: str):
    import mne

    reader = _EDF_CACHE.get(path)
    if reader is None:
        reader = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        _EDF_CACHE[path] = reader
    _EDF_CACHE.move_to_end(path)
    while len(_EDF_CACHE) > _READER_CACHE_SIZE:
        _, old = _EDF_CACHE.popitem(last=False)
        old.close()
    return reader


def close_cached_readers():
    """Close cached recording handles, for example before removing a temporary dataset."""
    for cache in (_H5_CACHE, _EDF_CACHE):
        while cache:
            _, reader = cache.popitem()
            reader.close()


atexit.register(close_cached_readers)


def _metadata(row, row_idx, item_idx, plan_item, valid_samples, n_samples):
    data = row._asdict() if hasattr(row, "_asdict") else dict(vars(row))
    data.update(
        {
            "row_idx": int(row_idx),
            "item_idx": int(item_idx),
            "is_augmented": plan_item is not None,
            "augmentation": json.dumps(_json_ready(plan_item or []), sort_keys=True),
            "valid_samples": int(valid_samples),
            "padding_samples": int(n_samples - valid_samples),
        }
    )
    return _json_ready(data)


def _store_cache_payload(rows, mode):
    records = [_window_record(row) for row in rows]
    windows_hash, row_count = _hash_sequence(records)
    return {
        "version": _CACHE_VERSION,
        "mode": mode,
        "store": "shared_windows",
        "row_count": row_count,
        "windows_hash": windows_hash,
    }


def _window_key(row) -> Tuple[Any, ...]:
    record = _window_identity(row)
    return (
        record.get("window_id"),
        record["signal_path"],
        record["signal_format"],
        record["start_s"],
        record["end_s"],
        record["window_length_s"],
        record["valid_duration_s"],
        record["label"],
    )


def _window_identity(row) -> Dict[str, Any]:
    path = _signal_path(row)
    return {
        "window_id": _none_if_nan(getattr(row, "window_id", None)),
        "signal_path": path,
        "signal_format": _signal_format(row, path),
        "start_s": float(getattr(row, "start_s")),
        "end_s": float(getattr(row, "end_s")),
        "window_length_s": float(getattr(row, "window_length_s", row.end_s - row.start_s)),
        "valid_duration_s": float(getattr(row, "valid_duration_s", row.end_s - row.start_s)),
        "label": _label_to_int(getattr(row, "label", "background")),
    }


def _window_record(row) -> Dict[str, Any]:
    record = _window_identity(row)
    path = record["signal_path"]
    try:
        stat = Path(path).stat()
    except OSError:
        record["signal_size"] = None
        record["signal_mtime_ns"] = None
    else:
        record["signal_size"] = int(stat.st_size)
        record["signal_mtime_ns"] = int(stat.st_mtime_ns)
    return record


def _write_manifest(store_dir, mode, signature, payload, files):
    _write_json(
        store_dir / "manifest.json",
        {"version": _CACHE_VERSION, "mode": mode, "signature": signature, "payload": payload, "files": files},
    )


def _manifest_valid(path: Path, mode: str, signature: str) -> bool:
    if not path.exists():
        return False
    try:
        manifest = json.loads(path.read_text())
    except Exception:
        return False
    if manifest.get("version") != _CACHE_VERSION or manifest.get("mode") != mode:
        return False
    if manifest.get("signature") != signature:
        return False
    return all(Path(file_path).exists() for file_path in manifest.get("files", {}).values())


def _cache_root(cache_dir, rows):
    if cache_dir is not None:
        return Path(cache_dir).expanduser().resolve()
    roots = []
    for row in rows:
        value = getattr(row, "dataset_root", None)
        if value and str(value) != "nan":
            roots.append(str(value))
    return (Path(os.path.commonpath(roots)) if roots else Path.cwd()) / "materialized_cache"


def _materialize_mode(value):
    if value is None or str(value).lower() in {"none", "lazy"}:
        return None
    mode = str(value).lower()
    if mode not in {"memory", "disk", "memmap"}:
        raise ValueError("materialize must be one of None, 'memory', 'disk', or 'memmap'")
    return mode


def _label_to_int(label: Any) -> int:
    labels = {"0": 0, "background": 0, "1": 1, "sz": 1, "seizure": 1, "mixed": -100}
    key = str(label).strip().lower()
    if key not in labels:
        raise ValueError(f"Unknown window label {label!r}")
    return labels[key]


def _seed_worker(worker_id: int) -> None:
    close_cached_readers()  # Do not share inherited recording handles across forked workers.
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def _stable_seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def _hash_json(value) -> str:
    return hashlib.sha256(json.dumps(_json_ready(value), sort_keys=True).encode("utf-8")).hexdigest()


def _hash_sequence(values) -> tuple:
    digest = hashlib.sha256()
    count = 0
    for value in values:
        digest.update(json.dumps(_json_ready(value), sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
        count += 1
    return digest.hexdigest(), count


def _safe_name(*parts) -> str:
    return "_".join("".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(p)) for p in parts)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(_json_ready(value), indent=2, sort_keys=True))


def _none_if_nan(value):
    if value is None:
        return None
    if isinstance(value, float) and not np.isfinite(value):
        return None
    text = str(value)
    return None if text == "nan" else value


def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value
