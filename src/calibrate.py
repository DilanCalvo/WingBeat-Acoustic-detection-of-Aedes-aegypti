# Pick the decision threshold on a noisy validation set.

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from . import config
from .evaluate import MODEL_KEYS, MODEL_LABELS, evaluate_condition, score_all, trivial_f1


def pick_thresholds(y, proba) -> dict:
    # The two operating points, both chosen on validation data only.
    y = np.asarray(y)
    proba = np.asarray(proba)

    # precision_recall_curve returns thresholds aligned with p[:-1] / r[:-1].
    p, r, thr = precision_recall_curve(y, proba)
    p, r = p[:-1], r[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.nan_to_num(2 * p * r / (p + r))

    best_f1 = int(np.argmax(f1))
    out = {
        "t_f1": float(thr[best_f1]),
        "val_f1_at_t_f1": float(f1[best_f1]),
    }

    # Highest precision among thresholds that still hit the target recall.
    ok = np.flatnonzero(r >= config.TARGET_RECALL)
    if len(ok):
        best_rec = int(ok[np.argmax(p[ok])])
        out["t_recall90"] = float(thr[best_rec])
        out["val_precision_at_t_recall90"] = float(p[best_rec])
        out["val_recall_at_t_recall90"] = float(r[best_rec])
        out["target_recall_reached"] = True
    else:
        out["t_recall90"] = float(thr[int(np.argmax(r))])
        out["val_precision_at_t_recall90"] = float(p[int(np.argmax(r))])
        out["val_recall_at_t_recall90"] = float(r.max())
        out["target_recall_reached"] = False
    return out


def calibrate(which: str = "both", rescore: bool = False) -> dict:
    # Score the noisy val set and pick thresholds. Writes results/calibration.json.
    if not config.HYBRID_VAL_MANIFEST.exists():
        raise FileNotFoundError(
            f"No noisy validation set at {config.HYBRID_VAL_MANIFEST}. Run:\n"
            f"    python -m src.build_hybrid_testset --split val"
        )
    val = pd.read_csv(config.HYBRID_VAL_MANIFEST)

    # The calibration data must not overlap the test set.
    if config.HYBRID_MANIFEST.exists():
        test_segs = set(pd.read_csv(config.HYBRID_MANIFEST)["source_segment_path"])
        assert not (set(val["source_segment_path"]) & test_segs), (
            "LEAKAGE: a validation segment is also in the test set, so the calibrated "
            "threshold would be tuned on test data."
        )

    pred = score_all(val, which, rescore=rescore, cache=config.VAL_PREDICTIONS)
    models = [m for m in ("cnn", "cnn_aug", "baseline") if MODEL_KEYS[m] in pred.columns]

    out: dict = {
        "chosen_on": "noisy validation set (val segments + calibration-half CR noise)",
        "n_val_rows": int(len(pred)),
        "target_recall": config.TARGET_RECALL,
        "models": {},
    }

    print(f"\nCalibrating on {len(pred)} noisy validation rows "
          f"({pred.condition.nunique()} conditions, noise the models never trained on).\n")

    for model in models:
        # Pool all conditions: a real detector doesn't know the SNR of its input.
        thresholds = pick_thresholds(pred["label"].values, pred[MODEL_KEYS[model]].values)
        out["models"][model] = thresholds

        print(f"--- {MODEL_LABELS[model]} ---")
        print(f"  t*_f1        = {thresholds['t_f1']:.4f}   (F1 on val: "
              f"{thresholds['val_f1_at_t_f1']:.3f})")
        note = "" if thresholds["target_recall_reached"] else "  [target NOT reached!]"
        print(f"  t*_recall90  = {thresholds['t_recall90']:.4f}   (recall "
              f"{thresholds['val_recall_at_t_recall90']:.3f}, precision "
              f"{thresholds['val_precision_at_t_recall90']:.3f}){note}")

        # Check the threshold isn't just collapsing to "everything is a mosquito".
        m = evaluate_condition(pred["label"].values, pred[MODEL_KEYS[model]].values,
                               thr=thresholds["t_f1"])
        triv = trivial_f1(pred["label"].values)
        edge = thresholds["val_f1_at_t_f1"] - triv
        if edge < 0.03:
            verdict = "DEGENERATE: recalibrating just pushes it to answer 'aedes' to everything"
        elif m["mcc"] < 0.30:
            verdict = "WEAK: it gains something, but MCC stays on the floor"
        else:
            verdict = "REAL: recalibrating does rescue it"
        thresholds["val_trivial_f1"] = triv
        thresholds["val_edge_over_trivial"] = edge
        thresholds["val_mcc_at_t_f1"] = m["mcc"]
        thresholds["verdict"] = verdict
        print(f"  on val @ t*_f1: recall {m['recall']:.3f}  precision {m['precision']:.3f}  "
              f"MCC {m['mcc']:.3f}")
        print(f"  F1 {thresholds['val_f1_at_t_f1']:.3f} vs trivial floor {triv:.3f} "
              f"({edge:+.3f})  ->  {verdict}\n")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CALIBRATION_PATH.write_text(json.dumps(out, indent=2))
    print(f"Thresholds -> {config.CALIBRATION_PATH}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the decision threshold on noisy val")
    parser.add_argument("--model", choices=("baseline", "cnn", "cnn_aug", "both", "all"),
                        default="all")
    parser.add_argument("--rescore", action="store_true", help="ignore the probability cache")
    args = parser.parse_args()
    calibrate(args.model, args.rescore)


if __name__ == "__main__":
    main()
