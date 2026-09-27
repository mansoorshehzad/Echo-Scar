"""Shared AHA mappings and temporal summaries for the two feature extractors."""

import numpy as np
from scipy.signal import periodogram

VIEW_SEGMENTS = {
    "a2c": (1, 4, 7, 10, 13, 15),
    "a3c": (2, 5, 8, 11, 13, 15),
    "a4c": (3, 6, 9, 12, 14, 16),
}

# From apex to base. The apical cap (17) is omitted from classification.
WALL_SEGMENTS = {
    "a2c": ((13, 7, 1), (15, 10, 4)),
    "a3c": ((13, 8, 2), (15, 11, 5)),
    "a4c": ((14, 9, 3), (16, 12, 6)),
}

SUMMARY_NAMES = ("es", "ed", "rms", "spectral_entropy")


def phase_indices(segmentation: np.ndarray, myocardium_label: int) -> tuple[int, int]:
    """Return ES and ED frame indices from a segmentation in T x H x W order.

    The final analysis selected the maximum and minimum myocardial areas
    once per video, then used those frames for every feature in that video.
    """
    if segmentation.ndim != 3 or segmentation.shape[0] < 2:
        raise ValueError("Expected at least two segmentation frames in T x H x W order")
    areas = np.sum(segmentation == myocardium_label, axis=(1, 2))
    if not np.any(areas):
        raise ValueError("No myocardium pixels were found")
    if np.ptp(areas) == 0:
        raise ValueError("Myocardial area does not vary across frames; ES/ED cannot be assigned")
    return int(np.argmax(areas)), int(np.argmin(areas))


def summarize(features: np.ndarray, es: int, ed: int, frame_rate: float = 30.0) -> dict[str, np.ndarray]:
    """Summarize T x F features using the same ES/ED frames for every column.

    Missing radiomics measurements are filled with each feature's observed
    mean for RMS and entropy. A missing value at ES/ED stays missing and can
    be imputed within the training fold of the classifier.
    """
    values = np.asarray(features, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("Expected a T x F matrix with at least two frames")
    if not (0 <= es < len(values) and 0 <= ed < len(values)):
        raise ValueError("ES/ED indices are outside the video")
    if frame_rate <= 0:
        raise ValueError("Frame rate must be positive")

    valid = np.isfinite(values)
    count = valid.sum(axis=0)
    means = np.divide(np.where(valid, values, 0).sum(axis=0), count,
                      out=np.zeros(values.shape[1]), where=count > 0)
    filled = np.where(valid, values, means[None, :])
    rms = np.sqrt(np.mean(filled ** 2, axis=0))
    _, power = periodogram(filled - filled.mean(axis=0), fs=frame_rate, axis=0)
    totals = power.sum(axis=0)
    probabilities = np.divide(power, totals[None, :], out=np.zeros_like(power),
                              where=totals[None, :] > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0, probabilities * np.log2(probabilities), 0)
    entropy = -terms.sum(axis=0)
    return {"es": values[es].copy(), "ed": values[ed].copy(),
            "rms": rms, "spectral_entropy": entropy}
