# Stage 4a: RandomForest baseline on MFCC features.

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from . import config
from .data_split import grouped_split


def load_split():
    # Load the MFCC features and produce the grouped split.
    x = np.load(config.MFCC_MATRIX_PATH)
    y = np.load(config.LABELS_PATH)
    groups = np.load(config.GROUPS_PATH)
    folders = np.load(config.FOLDERS_PATH, allow_pickle=True)
    train_idx, val_idx, test_idx = grouped_split(groups, folders)
    return x, y, folders, train_idx, val_idx, test_idx


def train_random_forest(x_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    # Fit a RandomForest with balanced class weights.
    clf = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        class_weight=config.RF_CLASS_WEIGHT,
        random_state=config.SEED,
        n_jobs=-1,
    )
    clf.fit(x_train, y_train)
    return clf


def false_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # FN / (FN + TP): fraction of present mosquitoes we missed.
    cm = confusion_matrix(y_true, y_pred, labels=[config.LABEL_NON_AEDES, config.LABEL_AEDES])
    tp = cm[1, 1]
    fn = cm[1, 0]
    return float(fn / (fn + tp)) if (fn + tp) else 0.0


def evaluate(clf, x_test, y_test) -> dict:
    # Clean-condition metrics for the baseline.
    y_pred = clf.predict(x_test)
    metrics = {
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "false_negative_rate": false_negative_rate(y_test, y_pred),
        "n_test": int(len(y_test)),
        "n_test_positives": int(y_test.sum()),
    }
    print("\n--- RandomForest baseline (clean test) ---")
    print(classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, zero_division=0))
    print(f"False-negative rate (missed mosquitoes): {metrics['false_negative_rate']:.3f}")
    return metrics


def per_folder_report(y_true, y_pred, folders_test) -> None:
    # Per-species error breakdown, to see if it separates species or just datasets.
    import pandas as pd

    df = pd.DataFrame({"folder": folders_test, "y": y_true, "pred": y_pred})
    df["dataset"] = df.folder.str.split("/").str[0]

    print("\n--- per source_folder (test) ---")
    rows = df.groupby("folder").apply(
        lambda g: pd.Series({
            "n": len(g),
            "label": int(g.y.iloc[0]) if g.y.nunique() == 1 else -1,
            "accuracy": round((g.y == g.pred).mean(), 3),
        }),
        include_groups=False,
    )
    print(rows.sort_values(["label", "accuracy"]).to_string())

    neg = df[df.y == 0]
    if not neg.empty:
        print("\n--- false-positive rate by dataset (negatives only) ---")
        for ds, sub in neg.groupby("dataset"):
            print(f"  {ds:<12} n={len(sub):>5}  FP rate={(sub.pred == 1).mean():.3f}")
        print("  (a much lower FP rate on a foreign dataset = domain shortcut, not species)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RandomForest baseline")
    parser.add_argument("--out", default=str(config.BASELINE_MODEL_PATH), help="output .pkl path")
    args = parser.parse_args()

    x, y, folders, train_idx, _val_idx, test_idx = load_split()
    clf = train_random_forest(x[train_idx], y[train_idx])
    metrics = evaluate(clf, x[test_idx], y[test_idx])
    per_folder_report(y[test_idx], clf.predict(x[test_idx]), folders[test_idx])

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # compress=3: ~65 MB uncompressed vs ~15 MB, so we can commit it without LFS.
    joblib.dump(clf, args.out, compress=3)
    metrics_path = config.RESULTS_DIR / "baseline_clean_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nModel  -> {args.out}")
    print(f"Metrics-> {metrics_path}")


if __name__ == "__main__":
    main()
