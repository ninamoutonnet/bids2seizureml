"""Deterministic EEG normalisation and augmentation utilities."""

import hashlib
from typing import Any, Dict, Optional, Sequence

import numpy as np


def normalise_eeg(x: np.ndarray, strategy: Optional[str] = None) -> np.ndarray:
    """Normalise valid samples of one channels x samples window, without fitted statistics."""

    if strategy is None or str(strategy).lower() in {"none", ""}:
        return x.astype(np.float32, copy=False)

    key = str(strategy).lower()
    if key == "zscore":
        mean = x.mean(axis=1, keepdims=True)
        std = np.maximum(x.std(axis=1, keepdims=True), 1e-8)
        return ((x - mean) / std).astype(np.float32, copy=False)

    if key == "global_zscore":
        mean = float(x.mean())
        std = max(float(x.std()), 1e-8)
        return ((x - mean) / std).astype(np.float32, copy=False)

    if key == "minmax":
        x_min = x.min(axis=1, keepdims=True)
        x_max = x.max(axis=1, keepdims=True)
        denom = np.maximum(x_max - x_min, 1e-8)
        return (((x - x_min) / denom) * 2.0 - 1.0).astype(np.float32, copy=False)

    raise ValueError("normalisation must be one of None, 'zscore', 'global_zscore', or 'minmax'")


def gaussian_noise(x: np.ndarray, rng: np.random.RandomState, snr_db: float = 20.0) -> np.ndarray:
    """Add channel-wise noise at the requested SNR, including for signals stored in volts."""
    if not np.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    signal_power = np.mean(np.square(x), axis=1, keepdims=True)
    noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape)
    return (x + noise).astype(np.float32, copy=False)


def amplitude_scaling(
    x: np.ndarray,
    rng: np.random.RandomState,
    min_scale: float = 0.8,
    max_scale: float = 1.2,
) -> np.ndarray:
    if not (np.isfinite(min_scale) and np.isfinite(max_scale) and 0 < min_scale <= max_scale):
        raise ValueError("Require 0 < min_scale <= max_scale")
    scale = rng.uniform(float(min_scale), float(max_scale), size=(x.shape[0], 1))
    return (x * scale).astype(np.float32, copy=False)


def time_shift(x: np.ndarray, rng: np.random.RandomState, max_shift_fraction: float = 0.10) -> np.ndarray:
    """Circularly shift all channels together; samples wrap around the window."""
    fraction = _probability(max_shift_fraction, "max_shift_fraction")
    max_shift = int(round(fraction * x.shape[1]))
    if max_shift <= 0:
        return x.astype(np.float32, copy=True)
    shift = int(rng.randint(-max_shift, max_shift + 1))
    return np.roll(x, shift=shift, axis=1).astype(np.float32, copy=False)


def channel_dropout(
    x: np.ndarray,
    rng: np.random.RandomState,
    channel_dropout_prob: float = 0.10,
    fill_value: float = 0.0,
) -> np.ndarray:
    prob = _probability(channel_dropout_prob, "channel_dropout_prob")
    out = x.copy()
    out[rng.rand(x.shape[0]) < prob, :] = float(fill_value)
    return out.astype(np.float32, copy=False)


AUGMENTATIONS = {
    "gaussian_noise": gaussian_noise,
    "amplitude_scaling": amplitude_scaling,
    "time_shift": time_shift,
    "channel_dropout": channel_dropout,
}


def build_augmentation_plan(rows: Sequence[Any], config: Optional[Dict[str, Any]]):
    """Keep originals and append fixed, reproducible copies of the requested labels.

    Multiple strategies are applied in their configured order to each extra copy.
    The same copy is returned on every access, including across training epochs.
    """

    if not config or config.get("enabled", True) is False:
        return [None for _ in rows]

    labels = _labels(config.get("labels", ["seizure"]))
    strategies = config.get("strategies", [])
    if isinstance(strategies, (str, dict)):
        strategies = [strategies]
    strategies = [_strategy_dict(strategy) for strategy in strategies]
    if not strategies:
        raise ValueError("augmentation requires at least one strategy")

    seed = int(config.get("seed", 42))
    copies = int(config.get("copies", config.get("n_augmented_per_sample", 1)))
    if copies < 0:
        raise ValueError("augmentation copies must be >= 0")

    plan = [None for _ in rows]
    for row_idx, row in enumerate(rows):
        if _row_label(row) not in labels:
            continue
        for copy_idx in range(copies):
            plan.append(
                [
                    _spec(strategy, seed, row_idx, copy_idx, strategy_idx)
                    for strategy_idx, strategy in enumerate(strategies)
                ]
            )
    return plan


def apply_augmentation_plan(x: np.ndarray, plan_item) -> np.ndarray:
    if not plan_item:
        return x.astype(np.float32, copy=True)
    out = x.astype(np.float32, copy=True)
    for spec in plan_item:
        rng = np.random.RandomState(int(spec["seed"]))
        out = AUGMENTATIONS[spec["name"]](out, rng=rng, **spec["params"])
    return out.astype(np.float32, copy=False)


def original_row_index(plan_item, item_idx: int) -> int:
    return item_idx if plan_item is None else int(plan_item[0]["row_idx"])


def _strategy_dict(strategy: Any) -> Dict[str, Any]:
    if isinstance(strategy, str):
        strategy = {"name": strategy}
    strategy = dict(strategy)
    name = str(strategy.pop("name")).lower()
    if name not in AUGMENTATIONS:
        raise ValueError(f"Unknown augmentation '{name}'. Available: {sorted(AUGMENTATIONS)}")
    return {"name": name, **strategy}


def _spec(strategy: Dict[str, Any], seed: int, row_idx: int, copy_idx: int, strategy_idx: int):
    params = {k: v for k, v in strategy.items() if k != "name"}
    return {
        "name": strategy["name"],
        "params": params,
        "seed": _stable_seed(seed, row_idx, copy_idx, strategy_idx, strategy["name"]),
        "row_idx": int(row_idx),
        "copy_idx": int(copy_idx),
        "strategy_idx": int(strategy_idx),
    }


def _labels(labels: Any) -> set:
    if labels is None or (isinstance(labels, str) and labels.lower() in {"all", "both"}):
        return {"background", "seizure"}
    if isinstance(labels, str):
        labels = [labels]
    return {_row_label(type("_LabelRow", (), {"label": label})()) for label in labels}


def _row_label(row: Any) -> str:
    value = getattr(row, "label", "background")
    labels = {"0": "background", "background": "background",
              "1": "seizure", "sz": "seizure", "seizure": "seizure", "mixed": "mixed"}
    key = str(value).strip().lower()
    if key not in labels:
        raise ValueError(f"Unknown window label {value!r}")
    return labels[key]


def _stable_seed(*parts: Any) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value
