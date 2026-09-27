"""Patient-grouped nested cross-validation for the Echo-SCAR feature arms.

Input labels have one row per video: column 0 is the video filename or case ID,
column 1 is a patient ID, and columns 2-18 are AHA segments 1-17 in order.
Identifiers are used only for matching and grouping; they are never features.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from temporal_features import SUMMARY_NAMES, VIEW_SEGMENTS


def case_id(name: str) -> str:
    """Remove only documented file suffixes; never guess an identifier from digits."""
    result = Path(str(name)).name
    for suffix in (".nii.gz", ".nii", ".npz", ".csv"):
        result = result.removesuffix(suffix)
    for suffix in ("_radiomics", "_0000"):
        result = result.removesuffix(suffix)
    return result


def load_labels(path: Path) -> pd.DataFrame:
    """Read the study's positional CMR label export and check join keys."""
    labels = pd.read_csv(path, dtype={0: str, 1: str})
    if labels.shape[1] < 19:
        raise ValueError("Labels need video ID, patient ID, and 17 ordered AHA columns")
    labels = labels.iloc[:, :19].copy()
    labels.columns = ["case_id", "patient_id"] + [f"seg{n:02d}" for n in range(1, 18)]
    labels["case_id"] = labels["case_id"].map(case_id)
    labels["patient_id"] = labels["patient_id"].astype("string").str.strip()
    if labels["case_id"].duplicated().any():
        raise ValueError("Duplicate case IDs after filename normalization")
    if labels["patient_id"].isna().any() or labels["patient_id"].eq("").any():
        raise ValueError("Every video needs a patient ID for grouped CV")
    for n in range(1, 18):
        column = f"seg{n:02d}"
        labels[column] = pd.to_numeric(labels[column], errors="coerce")
    return labels.set_index("case_id")


def load_radiomics(folder: Path) -> pd.DataFrame:
    """Pivot two-column per-video CSVs to one row per case."""
    rows = {}
    for path in sorted(folder.glob("*_radiomics.csv")):
        feature_table = pd.read_csv(path)
        if list(feature_table.columns) != ["feature", "value"]:
            raise ValueError(f"{path.name} needs feature,value columns")
        key = case_id(path.name)
        if key in rows or feature_table["feature"].duplicated().any():
            raise ValueError(f"Duplicate case or feature in {path}")
        rows[key] = pd.to_numeric(feature_table.set_index("feature")["value"], errors="coerce")
    if not rows:
        raise ValueError(f"No *_radiomics.csv files in {folder}")
    matrix = pd.DataFrame.from_dict(rows, orient="index").sort_index(axis=1)
    return matrix.apply(pd.to_numeric, errors="coerce")


def load_encoder(folder: Path) -> pd.DataFrame:
    """Read the four 512-channel summaries from each .npz file."""
    names = [f"{summary}_ch{channel:03d}" for summary in SUMMARY_NAMES for channel in range(512)]
    rows = {}
    for path in sorted(folder.glob("*.npz")):
        key = case_id(path.name)
        if key in rows:
            raise ValueError(f"Duplicate encoder case: {key}")
        with np.load(path) as archive:
            vector = np.asarray(archive["feature_vector"], dtype=float)
        if vector.shape != (2048,):
            raise ValueError(f"{path.name}: expected 2048 encoder features, got {vector.shape}")
        rows[key] = vector
    if not rows:
        raise ValueError(f"No .npz files in {folder}")
    return pd.DataFrame.from_dict(rows, orient="index", columns=names)


def segment_radiomics(radiomics: pd.DataFrame, segment: int) -> pd.DataFrame:
    """Select only features belonging to one AHA segment."""
    pattern = re.compile(rf"^\(Segment {segment}, original_[^)]+\)__(?:es|ed|rms|spectral_entropy)$")
    columns = [column for column in radiomics if pattern.match(str(column))]
    if not columns:
        raise ValueError(f"No radiomics features found for segment {segment}")
    return radiomics[columns]


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Remove highly correlated columns using only the fitted training fold."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        matrix = np.asarray(X, dtype=float)
        correlations = pd.DataFrame(matrix).corr().abs().fillna(0).to_numpy()
        keep = np.ones(matrix.shape[1], dtype=bool)
        for index in range(matrix.shape[1]):
            if keep[index] and np.any(correlations[index, :index][keep[:index]] > self.threshold):
                keep[index] = False
        self.keep_ = keep
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


class TopKFeatures(BaseEstimator, TransformerMixin):
    """Pick up to k radiomic features after fold-specific filtering."""

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, X, y):
        with np.errstate(divide="ignore", invalid="ignore"):
            scores, _ = f_classif(X, y)
        scores = np.nan_to_num(scores, nan=-np.inf)
        self.keep_ = np.sort(np.argsort(scores)[-min(self.k, X.shape[1]):])
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_]


def make_pipeline(arm: str, n_components: int) -> Pipeline:
    """Build the RF model specified in the paper; every fitted step is inside CV."""
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if arm == "radiomics":
        steps += [("correlation", CorrelationFilter()), ("select", TopKFeatures(k=50))]
    else:
        steps += [("scale", StandardScaler()), ("pca", PCA(n_components=n_components, random_state=42))]
    steps.append(("forest", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                    random_state=42, n_jobs=1)))
    return Pipeline(steps)


def valid_folds(y: np.ndarray, groups: np.ndarray, outer: int = 5, inner: int = 3):
    """Materialize grouped folds and check that every train/test side has both classes."""
    if len(np.unique(groups)) < outer:
        raise ValueError(f"Need at least {outer} distinct patients")
    splitter = StratifiedGroupKFold(n_splits=outer, shuffle=True, random_state=42)
    folds = list(splitter.split(np.zeros(len(y)), y, groups))
    for train, test in folds:
        if len(np.unique(y[train])) < 2 or len(np.unique(y[test])) < 2:
            raise ValueError("A grouped outer fold has only one class; more patients are needed")
        inner_splitter = StratifiedGroupKFold(n_splits=inner, shuffle=True, random_state=43)
        for inner_train, inner_test in inner_splitter.split(np.zeros(len(train)), y[train], groups[train]):
            if len(np.unique(y[train][inner_train])) < 2 or len(np.unique(y[train][inner_test])) < 2:
                raise ValueError("A grouped inner fold has only one class; more patients are needed")
    return folds


def nested_predictions(X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                       arm: str, outer_folds: int = 5, inner_folds: int = 3,
                       n_estimators: int = 300) -> pd.DataFrame:
    """Fit a new grid search inside each outer training set and return OOF scores."""
    y_array = y.to_numpy(dtype=int)
    group_array = groups.to_numpy(dtype=str)
    folds = valid_folds(y_array, group_array, outer_folds, inner_folds)
    predictions = np.full(len(y), np.nan)
    fold_numbers = np.full(len(y), -1)
    for fold_number, (train, test) in enumerate(folds):
        inner_splitter = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=43)
        inner_splits = list(inner_splitter.split(X.iloc[train], y.iloc[train], groups.iloc[train]))
        smallest_train = min(len(inner_train) for inner_train, _ in inner_splits)
        components = min(50, X.shape[1], smallest_train - 1)
        if arm != "radiomics" and components < 1:
            raise ValueError("Not enough training cases for PCA")
        pipeline = make_pipeline(arm, components)
        pipeline.set_params(forest__n_estimators=n_estimators)
        search = GridSearchCV(
            pipeline,
            {"forest__max_depth": [5, 10, None], "forest__min_samples_leaf": [2, 4]},
            cv=inner_splits, scoring="roc_auc", refit=True, n_jobs=1, error_score="raise",
        )
        search.fit(X.iloc[train], y.iloc[train])
        predictions[test] = search.predict_proba(X.iloc[test])[:, 1]
        fold_numbers[test] = fold_number
    if np.isnan(predictions).any():
        raise RuntimeError("Some cases did not receive an outer-fold prediction")
    return pd.DataFrame({"case_id": X.index, "patient_id": groups.values,
                         "y_true": y.values, "y_prob": predictions, "fold": fold_numbers})


def grouped_auc_ci(y: np.ndarray, probabilities: np.ndarray, groups: np.ndarray,
                   n_bootstrap: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap patients, preserving all videos from each sampled patient."""
    auc = float(roc_auc_score(y, probabilities))
    unique = np.unique(groups)
    positions = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_bootstrap):
        chosen = rng.choice(unique, len(unique), replace=True)
        indexes = np.concatenate([positions[group] for group in chosen])
        if len(np.unique(y[indexes])) == 2:
            samples.append(roc_auc_score(y[indexes], probabilities[indexes]))
    if len(samples) < max(20, n_bootstrap // 4):
        return auc, np.nan, np.nan
    low, high = np.percentile(samples, [2.5, 97.5])
    return auc, float(low), float(high)


def delong_paired(y: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    """Two-sided paired DeLong test on aligned video-level predictions."""
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if len(positive) < 2 or len(negative) < 2:
        return np.nan

    def placements(scores):
        comparisons = scores[positive, None] - scores[None, negative]
        wins = (comparisons > 0).astype(float) + .5 * (comparisons == 0)
        return wins.mean(axis=1), wins.mean(axis=0)

    a_pos, a_neg = placements(first)
    b_pos, b_neg = placements(second)
    covariance = (np.cov([a_pos, b_pos]) / len(positive) +
                  np.cov([a_neg, b_neg]) / len(negative))
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    if variance <= 0:
        return np.nan
    difference = roc_auc_score(y, first) - roc_auc_score(y, second)
    return float(2 * norm.sf(abs(difference) / np.sqrt(variance)))


def visible_scar_target(labels: pd.DataFrame, view: str) -> pd.Series:
    """Positive if any visible segment has scar; negative only if all are known."""
    visible = labels[[f"seg{segment:02d}" for segment in VIEW_SEGMENTS[view]]]
    positive = visible.ge(1).any(axis=1)
    fully_observed = visible.notna().all(axis=1)
    target = pd.Series(np.nan, index=visible.index, dtype=float)
    target.loc[fully_observed] = 0.0
    target.loc[positive] = 1.0
    return target


def evaluate_view(view: str, radiomics_dir: Path, encoder_dir: Path, labels_csv: Path,
                  output_dir: Path, n_bootstrap: int = 1000,
                  n_estimators: int = 300) -> pd.DataFrame:
    """Evaluate six segment targets and the view-level visible-scar target."""
    labels = load_labels(labels_csv)
    radiomics = load_radiomics(radiomics_dir)
    encoder = load_encoder(encoder_dir)
    common = labels.index.intersection(radiomics.index).intersection(encoder.index).sort_values()
    if len(common) == 0:
        raise ValueError("No case IDs match across labels, radiomics, and encoder features")
    if len(common) < len(labels):
        print(f"{view}: matched {len(common)}/{len(labels)} labeled videos")
    labels, radiomics, encoder = labels.loc[common], radiomics.loc[common], encoder.loc[common]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    def run_target(target: str, raw_label: pd.Series, use_radiomics: bool):
        numeric = pd.to_numeric(raw_label, errors="coerce")
        valid = numeric.notna()
        cases = labels.index[valid]
        y = (numeric.loc[cases] >= 1).astype(int)
        groups = labels.loc[cases, "patient_id"]
        if y.nunique() != 2:
            raise ValueError(f"{view}/{target}: only one label class")
        matrices = {"encoder": encoder.loc[cases]}
        if use_radiomics:
            segment = int(target.removeprefix("seg"))
            rad = segment_radiomics(radiomics.loc[cases], segment)
            matrices = {"radiomics": rad, "encoder": encoder.loc[cases],
                        "fusion": pd.concat([rad, encoder.loc[cases]], axis=1)}
        target_dir = output_dir / target
        target_dir.mkdir(exist_ok=True)
        predictions = {}
        for arm, matrix in matrices.items():
            oof = nested_predictions(matrix, y, groups, arm, n_estimators=n_estimators)
            oof.to_csv(target_dir / f"{arm}_oof.csv", index=False)
            probabilities = oof["y_prob"].to_numpy()
            auc, low, high = grouped_auc_ci(y.to_numpy(), probabilities,
                                            groups.to_numpy(), n_bootstrap)
            row = {"view": view, "target": target, "arm": arm, "n_videos": len(y),
                   "n_patients": groups.nunique(), "n_positive": int(y.sum()),
                   "auc": auc, "auc_ci_low": low, "auc_ci_high": high}
            # A fixed threshold can be applied to held-out predictions without
            # selecting an optimistic threshold on those same predictions.
            row["sensitivity_at_0_5"] = float(np.mean(probabilities[y.to_numpy() == 1] >= .5))
            row["specificity_at_0_5"] = float(np.mean(probabilities[y.to_numpy() == 0] < .5))
            rows.append(row)
            predictions[arm] = probabilities
        if use_radiomics:
            tests = {"radiomics_vs_encoder": delong_paired(y.to_numpy(), predictions["radiomics"],
                                                            predictions["encoder"]),
                     "radiomics_vs_fusion": delong_paired(y.to_numpy(), predictions["radiomics"],
                                                           predictions["fusion"])}
            (target_dir / "delong.json").write_text(json.dumps(tests, indent=2) + "\n")

    for segment in VIEW_SEGMENTS[view]:
        run_target(f"seg{segment:02d}", labels[f"seg{segment:02d}"], True)

    run_target("visible_scar", visible_scar_target(labels, view), False)
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def write_paper_tables(summary: pd.DataFrame, results_dir: Path) -> None:
    """Write compact tables corresponding to manuscript Tables 1 and 2."""
    segment_rows = []
    for view in VIEW_SEGMENTS:
        for segment in VIEW_SEGMENTS[view]:
            target = f"seg{segment:02d}"
            subset = summary.loc[(summary["view"] == view) & (summary["target"] == target)]
            if subset.empty:
                continue
            auc = dict(zip(subset["arm"], subset["auc"]))
            tests_path = results_dir / view / target / "delong.json"
            tests = json.loads(tests_path.read_text())
            segment_rows.append({"view": view.upper(), "segment": segment,
                                 "radiomics_auc": auc["radiomics"],
                                 "encoder_auc": auc["encoder"],
                                 "fusion_auc": auc["fusion"],
                                 "radiomics_vs_encoder_p": tests["radiomics_vs_encoder"],
                                 "radiomics_vs_fusion_p": tests["radiomics_vs_fusion"]})
    pd.DataFrame(segment_rows).to_csv(results_dir / "table1_segment_auc.csv", index=False)
    view_rows = summary.loc[summary["target"] == "visible_scar", [
        "view", "n_videos", "n_patients", "auc", "auc_ci_low", "auc_ci_high",
        "sensitivity_at_0_5", "specificity_at_0_5",
    ]].copy()
    view_rows["view"] = view_rows["view"].str.upper()
    view_rows.to_csv(results_dir / "table2_view_screening.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radiomics-dir", type=Path, required=True,
                        help="Parent directory containing a2c/, a3c/, a4c/")
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", nargs="+", choices=VIEW_SEGMENTS, default=list(VIEW_SEGMENTS))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()
    summaries = []
    for view in args.views:
        summary = evaluate_view(view, args.radiomics_dir / view, args.encoder_dir / view,
                                args.labels_dir / f"{view}.csv", args.output_dir / view,
                                args.bootstrap)
        summaries.append(summary)
    combined = pd.concat(summaries)
    combined.to_csv(args.output_dir / "all_views_summary.csv", index=False)
    write_paper_tables(combined, args.output_dir)


if __name__ == "__main__":
    main()
