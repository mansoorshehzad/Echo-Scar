# Echo-SCAR scripts

Keep these four Python files in the same directory and run the commands below from that directory.

| File | Use |
| --- | --- |
| `extract_radiomics.py` | Extract temporal radiomics from cine images and existing myocardium masks. |
| `extract_encoder_features.py` | Extract temporal encoder features using an nnU-Net checkpoint and the same masks. |
| `evaluate_scar.py` | Evaluate radiomics, encoder, and combined features against segment labels. |
| `temporal_features.py` | Shared AHA mappings and temporal calculations imported by the other scripts. |

## Installation

Use Python 3.12 and Git. Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "numpy>=1.24" "pandas>=2.0" "scipy>=1.10" "scikit-learn>=1.3" "nibabel>=5" "SimpleITK>=2.3" "torch>=2.0" "dynamic-network-architectures>=0.3"

```

For evaluation from existing feature files, install only NumPy, pandas, SciPy, and scikit-learn.

## Input files

Arrange inputs by view:

```text
data/
  cines/
    a2c/
    a3c/
    a4c/
      case001_0000.nii.gz
  segmentations/
    a2c/
    a3c/
    a4c/
      case001.nii.gz
  labels/
    a2c.csv
    a3c.csv
    a4c.csv
  nnUNet_results/
    Dataset001_CamusA2C/
    Dataset007_MIMICA3CPlus/
    Dataset002_CamusA4C/
```

Produce the segmentation masks with nnU-Net before running the extraction scripts. A cine named `case001_0000.nii.gz` requires a mask named `case001.nii.gz`. A cine named `case001.nii.gz` uses that same mask filename in the segmentation directory.

Both volumes must be 3D, have matching dimensions and frame order, and contain no singleton axes. Use H × W × T or T × H × W arrays with T smaller than both image dimensions; the scripts identify the time axis by its size. Every frame should contain myocardium, and myocardial area must vary across the video.

| View | Myocardium label | Default checkpoint dataset |
| --- | --- | --- |
| `a2c` | `2` | `Dataset001_CamusA2C` |
| `a3c` | `1` | `Dataset007_MIMICA3CPlus` |
| `a4c` | `2` | `Dataset002_CamusA4C` |

Place each checkpoint at:

```text
data/nnUNet_results/<dataset>/nnUNetTrainer__nnUNetPlans__2d/fold_all/checkpoint_final.pth
```

### Label CSVs

Each CSV must have a header and one row per video. The first 19 columns must be in this order:

1. Video filename or case ID.
2. Patient ID, repeated consistently across all videos belonging to that patient.
3. Seventeen CMR scores, ordered by AHA segment 1 through 17.

For example:

```csv
case_id,patient_id,seg01,seg02,seg03,seg04,seg05,seg06,seg07,seg08,seg09,seg10,seg11,seg12,seg13,seg14,seg15,seg16,seg17
case001,patient001,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0
case002,patient002,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
```

The loader uses column positions; header names may differ. Enter scores from 0 through 5 and leave unknown scores blank. Scores of 1 or greater are positive. Each case ID must be unique after filename suffixes are removed. The suffixes `.nii.gz`, `.nii`, `.npz`, `.csv`, `_radiomics`, and `_0000` are removed during matching.

## Extract radiomics

Run one view at a time:

```bash
python extract_radiomics.py \
  --view a4c \
  --cine-dir data/cines/a4c \
  --seg-dir data/segmentations/a4c \
  --myocardium-label 2 \
  --output-dir features/radiomics/a4c
```

Repeat for A2C and A3C, changing the view, directories, and myocardium label according to the table above.

Optional arguments:

- `--frame-rate 30`: frame rate in frames per second; default `30`.
- `--apex-end top`: place the apex at the top of the image. Use `bottom` for the opposite orientation or `auto` to select it from the mask; default `auto`.

Each video produces `<case>_radiomics.csv` with `feature` and `value` columns. Feature names contain the AHA segment, PyRadiomics measurement, and one of `es`, `ed`, `rms`, or `spectral_entropy`.

## Extract encoder features

Use the same cine and segmentation directories:

```bash
python extract_encoder_features.py \
  --view a4c \
  --cine-dir data/cines/a4c \
  --seg-dir data/segmentations/a4c \
  --checkpoint-root data/nnUNet_results \
  --output-dir features/encoder/a4c \
  --device cpu
```

Repeat for A2C and A3C with their corresponding directories and views. Use `--device cuda` for a CUDA-enabled PyTorch installation. Set `--frame-rate` to the same value used for radiomics extraction.

Each video produces `<case>.npz` containing:

- `es`, `ed`, `rms`, `spectral_entropy`: four arrays of 512 features.
- `feature_vector`: the four arrays concatenated in that order, with 2,048 entries.
- `es_idx`, `ed_idx`: zero-based frame indices.

To override checkpoint settings, save a JSON file and pass its path with `--model-config`. Supported settings are `dataset`, `stages`, `channels`, `classes`, `myocardium_label`, `mean`, and `std`. Unspecified settings retain the selected view's defaults. For example:

```json
{
  "dataset": "Dataset007_MIMICA3CPlus",
  "myocardium_label": 1,
  "mean": 77.89,
  "std": 50.90
}
```

Use settings that match the checkpoint. The channel list must have one entry per stage and end in 512 channels.

## Run evaluation

After extraction, the feature directories should contain:

```text
features/
  radiomics/
    a2c/
    a3c/
    a4c/
  encoder/
    a2c/
    a3c/
    a4c/
```

Run all three views:

```bash
python evaluate_scar.py \
  --radiomics-dir features/radiomics \
  --encoder-dir features/encoder \
  --labels-dir data/labels \
  --output-dir results
```

Add `--views a4c` to evaluate only A4C, or list multiple views, such as `--views a2c a3c`. Set `--bootstrap` to change the number of patient bootstrap samples; default `1000`.

Evaluation includes cases with matching labels, radiomics files, and encoder files. It runs five outer folds and three inner folds grouped by patient. Each evaluated target must contain both classes in every training and validation fold.

Outputs include:

| Path under `results/` | Contents |
| --- | --- |
| `<view>/segXX/radiomics_oof.csv` | Held-out radiomics predictions and fold assignments. |
| `<view>/segXX/encoder_oof.csv` | Held-out encoder predictions and fold assignments. |
| `<view>/segXX/fusion_oof.csv` | Held-out combined-feature predictions and fold assignments. |
| `<view>/segXX/delong.json` | Paired AUC comparisons. |
| `<view>/visible_scar/encoder_oof.csv` | Held-out predictions for any scar in the view's visible segments. |
| `<view>/summary.csv` | Metrics for one view. |
| `all_views_summary.csv` | Combined metrics for the requested views. |
| `table1_segment_auc.csv` | Segment-level AUC comparisons. |
| `table2_view_screening.csv` | View-level AUC, confidence intervals, sensitivity, and specificity. |

Sensitivity and specificity use a probability cutoff of `0.5`.

## Shared temporal functions

The extraction scripts import `temporal_features.py` automatically. To use its functions directly, provide a segmentation array in T × H × W order and a feature array in T × F order:

```python
from temporal_features import phase_indices, summarize

es, ed = phase_indices(segmentation, myocardium_label=2)
summaries = summarize(frame_features, es, ed, frame_rate=30.0)
```

## Command help

```bash
python extract_radiomics.py --help
python extract_encoder_features.py --help
python evaluate_scar.py --help
```
