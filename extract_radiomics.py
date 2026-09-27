"""Extract per-segment PyRadiomics features from cine and segmentation volumes.

The script expects matching NIfTI videos and nnU-Net segmentation outputs.
The input volumes may be H x W x T or T x H x W; the time axis is inferred
from the image dimensions and must agree between image and mask.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from temporal_features import VIEW_SEGMENTS, WALL_SEGMENTS, SUMMARY_NAMES, phase_indices, summarize


def time_first(volume: np.ndarray) -> np.ndarray:
    """Convert a 3D cine or mask to T x H x W, rejecting ambiguous shapes."""
    if volume.ndim != 3 or min(volume.shape) < 2:
        raise ValueError(f"Expected a 3D volume without singleton axes, got {volume.shape}")
    if volume.shape[2] < min(volume.shape[:2]):
        return np.moveaxis(volume, 2, 0)
    if volume.shape[0] < min(volume.shape[1:]):
        return volume
    raise ValueError(f"Cannot infer time axis from shape {volume.shape}")


def assign_aha_segments(frame: np.ndarray, view: str, myocardium_label: int,
                        apex_end: str = "auto") -> np.ndarray:
    """Divide the segmented myocardium into six visible AHA regions.

    This preserves the geometric division used by the original extraction
    script: 15% apical cap, then apical/mid/basal wall bands at 33% and 66%.
    The apical cap is assigned 17 so it can be excluded from the analysis.
    Inspect the resulting masks before using them on a new acquisition set.
    """
    coords = np.argwhere(frame == myocardium_label)
    if len(coords) < 4:
        raise ValueError("Too few myocardium pixels to assign AHA segments")
    top, bottom = coords[:, 0].min(), coords[:, 0].max()
    if apex_end == "auto":
        band = max(1, int(round(0.08 * (bottom - top))))
        top_width = np.ptp(coords[coords[:, 0] <= top + band, 1])
        bottom_width = np.ptp(coords[coords[:, 0] >= bottom - band, 1])
        apex_end = "top" if top_width <= bottom_width else "bottom"
    if apex_end not in {"top", "bottom"}:
        raise ValueError("apex_end must be auto, top, or bottom")

    apex_y = top if apex_end == "top" else bottom
    apex = coords[coords[:, 0] == apex_y].mean(axis=0)
    base_band = coords[coords[:, 0] >= bottom - max(1, int(0.08 * (bottom - top)))] \
        if apex_end == "top" else coords[coords[:, 0] <= top + max(1, int(0.08 * (bottom - top)))]
    base = base_band.mean(axis=0)
    axis = base - apex
    length2 = float(axis @ axis)
    if length2 == 0:
        raise ValueError("Cannot determine myocardial long axis")

    displacement = coords - apex
    fraction = (displacement @ axis) / length2
    cross = displacement[:, 1] * axis[0] - displacement[:, 0] * axis[1]
    left, right = WALL_SEGMENTS[view]
    labels = np.where(cross > 0,
                      np.select([fraction < .15, fraction < .33, fraction < .66],
                                [17, left[0], left[1]], default=left[2]),
                      np.select([fraction < .15, fraction < .33, fraction < .66],
                                [17, right[0], right[1]], default=right[2]))
    output = np.zeros(frame.shape, dtype=np.uint8)
    output[coords[:, 0], coords[:, 1]] = labels
    return output


def make_extractor():
    """Configure the feature classes used in the radiomics arm."""
    from radiomics import featureextractor

    extractor = featureextractor.RadiomicsFeatureExtractor(force2D=True, force2Ddimension=0)
    extractor.disableAllFeatures()
    for feature_class in ("shape2D", "firstorder", "glcm", "glrlm", "gldm", "glszm", "ngtdm"):
        extractor.enableFeatureClassByName(feature_class)
    return extractor


def extract_case(cine: np.ndarray, segmentation: np.ndarray, view: str,
                 myocardium_label: int,
                 frame_rate: float = 30.0, apex_end: str = "auto") -> pd.DataFrame:
    """Return a two-column feature/value table for one cine video."""
    from SimpleITK import GetImageFromArray

    if cine.shape != segmentation.shape:
        raise ValueError(f"Cine shape {cine.shape} differs from segmentation {segmentation.shape}")
    es, ed = phase_indices(segmentation, myocardium_label)
    extractor = make_extractor()
    series: dict[tuple[int, str], np.ndarray] = {}
    for t, (image, mask) in enumerate(zip(cine, segmentation)):
        aha = assign_aha_segments(mask, view, myocardium_label, apex_end)
        image_itk = GetImageFromArray(image.astype(np.float32))
        mask_itk = GetImageFromArray(aha)
        for segment in VIEW_SEGMENTS[view]:
            if np.count_nonzero(aha == segment) < 4:
                continue
            try:
                result = extractor.execute(image_itk, mask_itk, label=segment)
            except (RuntimeError, ValueError):
                continue  # Too little valid myocardium in this frame.
            for name, value in result.items():
                if not name.startswith("original_") or "SumAverage" in name:
                    continue
                try:
                    numeric_value = np.asarray(value).item()
                    # Some PyRadiomics edge cases return complex values. They
                    # are not real image measurements, so leave them missing.
                    if np.iscomplexobj(numeric_value):
                        if abs(numeric_value.imag) > 1e-8:
                            continue
                        numeric_value = numeric_value.real
                    numeric = float(numeric_value)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(numeric):
                    continue
                series.setdefault((segment, name), np.full(len(cine), np.nan))[t] = numeric

    rows = []
    for (segment, name), values in sorted(series.items()):
        summary = summarize(values[:, None], es, ed, frame_rate)
        for kind in SUMMARY_NAMES:
            rows.append((f"(Segment {segment}, {name})__{kind}", summary[kind][0]))
    return pd.DataFrame(rows, columns=["feature", "value"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cine-dir", type=Path, required=True)
    parser.add_argument("--seg-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", choices=VIEW_SEGMENTS, required=True)
    parser.add_argument("--myocardium-label", type=int, required=True)
    parser.add_argument("--frame-rate", type=float, default=30.0)
    parser.add_argument("--apex-end", choices=("auto", "top", "bottom"), default="auto")
    args = parser.parse_args()
    import nibabel as nib

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.cine_dir.glob("*.nii.gz"))
    if not files:
        parser.error("No .nii.gz cines found")
    for cine_path in files:
        case_id = cine_path.name.removesuffix(".nii.gz").removesuffix("_0000")
        seg_path = args.seg_dir / f"{case_id}.nii.gz"
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing segmentation for {cine_path.name}: {seg_path}")
        cine = time_first(np.asarray(nib.load(cine_path).dataobj))
        segmentation = time_first(np.asarray(nib.load(seg_path).dataobj))
        features = extract_case(cine, segmentation, args.view, args.myocardium_label,
                                args.frame_rate, args.apex_end)
        if features.empty:
            raise ValueError(f"No features extracted for {case_id}")
        features.to_csv(args.output_dir / f"{case_id}_radiomics.csv", index=False)
        print(f"{case_id}: {len(features)} features")


if __name__ == "__main__":
    main()
