# bids2seizureml

bids2seizureml prepares EEG seizure datasets stored in [EEG-BIDS](https://bids-specification.readthedocs.io/en/stable/modality-specific-files/electroencephalography.html)
for machine learning. It takes a dataset from its converted recordings to the PyTorch data loaders of
each cross-validation fold, in four stages:

1. **Preprocessing.** Harmonises the recordings (montage, channels, sampling frequency, filtering) and
   writes them to a folder named after the settings, so that different configurations never overwrite
   each other and the original recordings are never modified.
2. **Segmentation.** Divides each recording into seizure and background intervals, then into
   fixed-length windows, using only the annotations and metadata: no EEG is read.
3. **Fold construction.** Assigns whole patients (or whole recordings, for patient-specific
   evaluation) to training, validation and test sets before any window is selected, then samples the
   windows of each set. No EEG is read.
4. **Data loading.** Reads the windows of each fold and returns PyTorch data loaders, with optional
   normalisation and augmentation.

Every intermediate result is a pandas dataframe, which can be inspected, saved or modified before the
next stage.

## Installation

Requires Python 3.11 or later. From the root of this repository:

```bash
python -m pip install .               # the package
python -m pip install ".[examples]"   # the package and what the example notebooks need
```

## Supported datasets

The package expects an EEG-BIDS dataset with:

- recordings in EDF format (`*_eeg.edf`), each with a JSON sidecar (`*_eeg.json`);
- annotations in `*_events.tsv` files, in which seizure events have a label starting with `sz`.

These are the conventions of [epilepsy2bids](https://github.com/esl-epfl/epilepsy2bids), which converts
the CHB-MIT, Siena, TUSZ and SeizeIT datasets to EEG-BIDS. The package has been used with the CHB-MIT,
Siena and TUSZ datasets converted by epilepsy2bids 0.0.7.

## Quick start

```python
from bids2seizureml.preprocessing import preprocess_bids
from bids2seizureml.segment import build_segments_dataframe, build_windowed_segments_dataframe
from bids2seizureml.folds import CVPlan, create_fold_dataframes
from bids2seizureml.loaders import build_dataloaders_from_folds

# 1. Preprocessing: written to data/chbmit/derivatives/preprocessing_<signature>/
derivative = preprocess_bids(
    "data/chbmit",
    montage="Longitudinal",
    notch_filter=60.0,
    target_sampling_freq=256.0,
    output_format="h5",
)

# 2. Segmentation: 4-second windows, each entirely seizure or entirely background
segments = build_segments_dataframe(str(derivative), centre="chbmit")
windows = build_windowed_segments_dataframe(segments, segment_length=4.0)

# 3. Fold construction: leave-one-patient-out, one background window per seizure window
plan = CVPlan(
    strategy="lopo",
    eval_strategy="balanced_windows",
    background_to_seizure_ratio=1.0,
    min_background_seizure_distance_s=30.0,
    min_background_recording_edge_distance_s=30.0,
)
folds = create_fold_dataframes(windows, plan)

# 4. Data loading: one training and one test loader per fold
loaders = build_dataloaders_from_folds(folds, batch_size=64)
x, y = next(iter(loaders["subject_chbmit-01"]["train"]))  # x: 64 x 18 x 1024, y: 0 or 1
```

## What each stage offers

**Preprocessing** (`preprocess_bids`)

- Channel labels are standardised (T7, T8, P7 and P8 become T3, T4, T5 and T6), including inside
  bipolar labels.
- Target montage: common average, Cz-referential or longitudinal bipolar (double banana). Only
  common-average recordings can be re-referenced; a recording that cannot be brought to the requested
  montage, or that lacks a requested channel, is excluded with a warning.
- Channel selection, resampling, notch filter and bandpass filter, applied in that order.
- Output in EDF or HDF5, with an updated JSON sidecar and the original `events.tsv`.
- Interrupted runs resume where they stopped, and recordings can be processed in parallel.

**Segmentation** (`build_segments_dataframe`, `build_windowed_segments_dataframe`,
`build_evaluation_windows_dataframe`)

- Overlapping or touching seizure annotations are merged; the rest of each recording is background.
- Windows have a fixed length, with separate overlaps for seizure and background windows, and record
  their distance to the nearest seizure and recording boundary.
- Evaluation windows cover each recording completely, and are labelled seizure, background or mixed.
- Several datasets can be combined with `build_segments_dataframe_multi`.

**Fold construction** (`create_fold_dataframes`, configured with `CVPlan`)

- Strategies: leave-one-patient-out (`lopo`), patient-independent k-fold (`kfold_patient`),
  leave-one-centre-out (`leave_one_centre_out`) and patient-specific evaluation (`patient_specific`),
  in which the recordings of each patient are split chronologically.
- All seizure windows are kept, and background windows are sampled per patient at a given ratio, with a
  fixed seed per patient and optional minimum distances to seizures and recording boundaries.
- Validation and test sets are either sampled in the same way (`balanced_windows`) or comprise every
  window of the held-out recordings (`complete_recordings`, the default).

**Data loading** (`build_dataloaders_from_folds`)

- Each window is a `channels x samples` tensor, labelled 0 (background), 1 (seizure) or -100 (mixed,
  ignored by `torch.nn.CrossEntropyLoss`).
- Windows are read when needed (default), or read once and shared by all folds, in memory
  (`materialize="memory"`) or on disk (`"disk"` or `"memmap"`).
- Per-window normalisation, and augmentation of the training set (Gaussian noise, amplitude scaling,
  time shift, channel dropout), reproducible across epochs and runs.

## Examples

The notebooks in [`examples/`](examples) run the whole pipeline on the Siena dataset, which is openly
available and small. They expect the converted datasets in `data/` at the root of this repository
(for example, `data/siena`); change the path at the top of each notebook otherwise.

| Notebook | Content |
| :-- | :-- |
| [`01_convert_to_bids.ipynb`](examples/01_convert_to_bids.ipynb) | Download links and conversion to EEG-BIDS with epilepsy2bids |
| [`02_preprocessing.ipynb`](examples/02_preprocessing.ipynb) | Preprocessing, and comparison of a recording before and after |
| [`03_windows_folds_training.ipynb`](examples/03_windows_folds_training.ipynb) | Windows, leave-one-patient-out folds, data loaders, and EEGNet trained on every fold |

## Notes

- Signals are stored and read in volts. HDF5 outputs store 64-bit floats and are about four times
  larger than the EDF recordings they come from.
- epilepsy2bids replaces missing electrodes with zeros before re-referencing. Such recordings keep
  their expected channel labels, so the preprocessing stage cannot detect them.

## Tests

```bash
python -m pip install ".[test]"
pytest
```

## Citation

Citation details will be added here.

## Licence

Released under the MIT licence. See [LICENSE](LICENSE).
