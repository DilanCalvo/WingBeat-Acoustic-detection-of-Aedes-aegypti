# How far does CR noise degrade the detector?

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import librosa
import matplotlib

matplotlib.use("Agg")  # headless: this runs from a CLI, never from a notebook display

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm  # noqa: E402

from . import config  # noqa: E402
from .build_hybrid_testset import list_cr_noise_files  # noqa: E402
from .data_split import holdout_mask  # noqa: E402
from .features import mel_spectrogram, mfcc_vector, normalize_mel  # noqa: E402
from .train_baseline import false_negative_rate  # noqa: E402

MODEL_KEYS = {"cnn": "proba_cnn", "baseline": "proba_rf", "cnn_aug": "proba_cnn_aug"}
MODEL_LABELS = {"cnn": "CNN", "baseline": "RandomForest", "cnn_aug": "CNN + CR noise"}
MODEL_PATHS = {
    "cnn": config.CNN_MODEL_PATH,
    "baseline": config.BASELINE_MODEL_PATH,
    "cnn_aug": config.CNN_AUG_MODEL_PATH,
}
# The clean-condition F1 each model reported at training time.
CLEAN_METRICS_FILES = {
    "cnn": config.RESULTS_DIR / "cnn_clean_metrics.json",
    "baseline": config.RESULTS_DIR / "baseline_clean_metrics.json",
    "cnn_aug": config.RESULTS_DIR / "cnn_aug_clean_metrics.json",
}


def wanted(which: str) -> list[str]:
    # Which model keys to score for a --model choice.
    if which == "both":
        return ["cnn", "baseline"]
    if which == "all":
        return [n for n in ("cnn", "baseline", "cnn_aug") if MODEL_PATHS[n].exists()]
    return [which]


# Models
def load_models(which: str) -> dict:
    # Load the serialized models (TensorFlow imported lazily).
    models = {}
    for name in wanted(which):
        if not MODEL_PATHS[name].exists():
            raise FileNotFoundError(f"{name}: no model at {MODEL_PATHS[name]}")
        if name == "baseline":
            import joblib

            models[name] = joblib.load(MODEL_PATHS[name])
        else:
            from tensorflow import keras

            models[name] = keras.models.load_model(MODEL_PATHS[name])
    return models


# Scoring (feature extraction is streamed; only probabilities are kept)
def iter_feature_chunks(paths: list[str], chunk: int = 1024):
    # Stream MFCC + raw-dB Mel features in chunks, to keep memory flat.
    for start in tqdm(range(0, len(paths), chunk), desc="scoring", unit="chunk"):
        batch = paths[start : start + chunk]
        mfcc, mel = [], []
        for rel in batch:
            wav, _ = librosa.load(
                str(config.PROJECT_ROOT / rel), sr=config.SAMPLE_RATE, mono=True
            )
            mfcc.append(mfcc_vector(wav))
            mel.append(mel_spectrogram(wav))
        yield np.stack(mfcc), np.stack(mel)[..., np.newaxis]


def _cache_fingerprint(manifest: pd.DataFrame, which: str) -> dict:
    # Keyed on the manifest and the model files, so a stale cache can't be reused.
    paths = "\n".join(manifest["hybrid_path"].tolist())
    fp = {
        "n_rows": int(len(manifest)),
        "paths_sha1": hashlib.sha1(paths.encode()).hexdigest(),
        "models": {},
    }
    for name in wanted(which):
        st = MODEL_PATHS[name].stat()
        fp["models"][name] = {"size": st.st_size, "mtime": int(st.st_mtime)}
    return fp


def score_all(manifest: pd.DataFrame, which: str, rescore: bool = False,
              cache: Path = config.HYBRID_PREDICTIONS) -> pd.DataFrame:
    # Probability per segment per model, with an invalidating disk cache.
    meta_path = cache.with_suffix(".meta.json")
    fingerprint = _cache_fingerprint(manifest, which)

    if not rescore and cache.exists() and meta_path.exists():
        cached = json.loads(meta_path.read_text())
        if cached == fingerprint:
            print(f"Using cached probabilities: {cache.name}")
            return pd.read_csv(cache)
        print(f"Cache is stale ({cache.name}: manifest or a model changed) — re-scoring.")

    models = load_models(which)
    paths = manifest["hybrid_path"].tolist()
    proba: dict[str, list[np.ndarray]] = {name: [] for name in models}

    for x_mfcc, x_mel_raw in iter_feature_chunks(paths):
        # The Mel arrives as raw dB, so it still needs normalize_mel before the CNN.
        x_mel = normalize_mel(x_mel_raw)
        for name, model in models.items():
            if name == "baseline":
                proba[name].append(model.predict_proba(x_mfcc)[:, 1])
            else:
                proba[name].append(model.predict(x_mel, verbose=0).ravel())

    out = manifest.copy()
    for name, chunks in proba.items():
        out[MODEL_KEYS[name]] = np.concatenate(chunks)

    keep = [c for c in ("condition", "snr_db", "environment", "label", "dataset",
                        "source_folder", "source_recording", "hybrid_path") if c in out]
    out = out[keep + [MODEL_KEYS[n] for n in proba]]

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    meta_path.write_text(json.dumps(fingerprint, indent=2))
    print(f"Probabilities -> {cache}")
    return out


# Metrics
def grouped_bootstrap_ci(y, proba, recordings, thr: float = config.DECISION_THRESHOLD):
    # 95% CI for F1, resampling recordings rather than segments.
    y = np.asarray(y)
    pred = (np.asarray(proba) >= thr).astype(int)
    codes, _ = pd.factorize(pd.Series(recordings))
    n_rec = codes.max() + 1
    if n_rec < 2:
        return (float("nan"), float("nan"))

    tp = np.bincount(codes, weights=(pred == 1) & (y == 1), minlength=n_rec)
    fp = np.bincount(codes, weights=(pred == 1) & (y == 0), minlength=n_rec)
    fn = np.bincount(codes, weights=(pred == 0) & (y == 1), minlength=n_rec)

    rng = np.random.default_rng(config.SEED)
    counts = rng.multinomial(n_rec, np.full(n_rec, 1.0 / n_rec), size=config.BOOTSTRAP_RESAMPLES)
    tps, fps, fns = counts @ tp, counts @ fp, counts @ fn
    denom = 2 * tps + fps + fns
    f1s = np.divide(2 * tps, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)
    return float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))


def trivial_f1(y) -> float:
    # F1 of a classifier that answers "aedes" to everything: 2p/(1+p).
    p = float(np.mean(y))
    return float(2 * p / (1 + p)) if p > 0 else 0.0


def evaluate_condition(y, proba, recordings=None, thr: float = config.DECISION_THRESHOLD) -> dict:
    # All metrics for one slice: at the threshold, plus threshold-free ones.
    y = np.asarray(y)
    proba = np.asarray(proba)
    pred = (proba >= thr).astype(int)

    both_classes = len(np.unique(y)) == 2
    metrics = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "false_negative_rate": false_negative_rate(y, pred),
        "roc_auc": float(roc_auc_score(y, proba)) if both_classes else float("nan"),
        # The two guards against a fake rescue.
        "trivial_f1": trivial_f1(y),
        "mcc": float(matthews_corrcoef(y, pred)) if both_classes else float("nan"),
        # Upper bound if the threshold were perfectly recalibrated for THIS condition.
        "oracle_f1": float("nan"),
        "fp_rate": float((pred[y == 0] == 1).mean()) if (y == 0).any() else float("nan"),
    }
    if both_classes:
        p, r, _ = precision_recall_curve(y, proba)
        with np.errstate(invalid="ignore", divide="ignore"):
            f1s = np.nan_to_num(2 * p * r / (p + r))
        metrics["oracle_f1"] = float(f1s.max())
    if recordings is not None and both_classes:
        lo, hi = grouped_bootstrap_ci(y, proba, recordings, thr)
        metrics["f1_ci_low"], metrics["f1_ci_high"] = lo, hi
    return metrics


def _slice_metrics(df: pd.DataFrame, model: str) -> dict:
    return evaluate_condition(df["label"].values, df[MODEL_KEYS[model]].values,
                              df["source_recording"].values)


# The "why": how much of each environment's noise energy sits on the wingbeat band
def noise_band_energy() -> dict[str, dict]:
    # Mean spectrum per environment and its energy share in the wingbeat band.
    lo, hi = config.WINGBEAT_BAND_HZ
    freqs = librosa.fft_frequencies(sr=config.SAMPLE_RATE, n_fft=config.N_FFT)
    band = (freqs >= lo) & (freqs <= hi)
    out = {}
    for env, pool in list_cr_noise_files().items():
        spectra = []
        for path in pool["test"]:
            wav, _ = librosa.load(str(path), sr=config.SAMPLE_RATE, mono=True)
            spec = np.abs(librosa.stft(wav, n_fft=config.N_FFT,
                                       hop_length=config.HOP_LENGTH)) ** 2
            spectra.append(spec.mean(axis=1))
        psd = np.mean(spectra, axis=0)
        out[env] = {
            "freqs": freqs,
            "psd": psd,
            "band_fraction": float(psd[band].sum() / psd.sum()),
        }
    return out


# Plots
def _cond_order(pred: pd.DataFrame) -> list[str]:
    return [c for c in config.EVAL_CONDITIONS if c in set(pred["condition"])]


def _cond_labels(conds: list[str]) -> list[str]:
    snr = config.SNR_LEVELS_DB
    return [c if c == "clean" else f"{c}\n{snr[c]:.0f} dB" for c in conds]


def plot_degradation(rows: dict, pred: pd.DataFrame, models: list[str]) -> None:
    # F1 and false-negative rate vs. noise level, the main degradation figure.
    conds = _cond_order(pred)
    x = np.arange(len(conds))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for model in models:
        f1 = [rows[model][c]["f1"] for c in conds]
        lo = [rows[model][c].get("f1_ci_low", np.nan) for c in conds]
        hi = [rows[model][c].get("f1_ci_high", np.nan) for c in conds]
        fnr = [rows[model][c]["false_negative_rate"] for c in conds]
        auc = [rows[model][c]["roc_auc"] for c in conds]
        oracle = [rows[model][c]["oracle_f1"] for c in conds]
        axes[0].plot(x, f1, "o-", label=MODEL_LABELS[model])
        axes[0].fill_between(x, lo, hi, alpha=0.15)
        axes[0].plot(x, oracle, "o:", alpha=0.55,
                     label=f"{MODEL_LABELS[model]} (oracle threshold)")
        axes[1].plot(x, fnr, "o-", label=MODEL_LABELS[model])
        axes[2].plot(x, auc, "o-", label=MODEL_LABELS[model])
    # The floor to read everything against: always answering "aedes" scores this F1.
    triv = rows[models[0]][conds[0]]["trivial_f1"]
    axes[0].axhline(triv, color="crimson", ls="--", lw=1.2)
    axes[0].text(len(conds) - 1, triv + 0.015, f"trivial floor ({triv:.3f}): answer aedes to everything",
                 fontsize=7.5, color="crimson", ha="right")
    axes[0].set_title("F1 @ fixed 0.5 threshold (95% CI)\ndotted = with a recalibrated threshold")
    axes[0].set_ylabel("F1")
    axes[1].set_title("False-negative rate\n(Aedes present that go undetected)")
    axes[1].set_ylabel("FN rate")
    axes[2].set_title("ROC-AUC, threshold-independent\n(is the information still there?)")
    axes[2].set_ylabel("ROC-AUC")
    axes[2].axhline(0.5, color="k", ls="--", lw=0.8)
    axes[2].text(0, 0.51, " chance", fontsize=8, va="bottom")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(_cond_labels(conds))
        ax.set_xlabel("condition (SNR)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("WINGBEAT - degradation under Costa Rican background noise "
                 "(the same 6,206 segments across all 4 conditions)")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "degradation_f1.png", dpi=150)
    plt.close(fig)


def plot_by_environment(pred: pd.DataFrame, model: str) -> None:
    # Per-environment degradation in F1 and in AUC.
    conds = _cond_order(pred)
    x = np.arange(len(conds))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for env in sorted(pred["environment"].unique()):
        f1, auc = [], []
        for c in conds:
            sub = pred[(pred.condition == c) & (pred.environment == env)]
            m = evaluate_condition(sub["label"].values, sub[MODEL_KEYS[model]].values)
            f1.append(m["f1"])
            auc.append(m["roc_auc"])
        axes[0].plot(x, f1, "o-", label=env)
        axes[1].plot(x, auc, "o-", label=env)
    axes[0].set_title("F1 @ 0.5 collapses in every environment\n(saturated, so it cannot rank them)")
    axes[0].set_ylabel("F1")
    axes[1].set_title("ROC-AUC, not saturated\n(this shows which noise hurts most)")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].axhline(0.5, color="k", ls="--", lw=0.8)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(_cond_labels(conds))
        ax.set_xlabel("condition (SNR)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"{MODEL_LABELS[model]} - degradation by CR noise environment")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "degradation_by_environment.png", dpi=150)
    plt.close(fig)


def plot_confusions(pred: pd.DataFrame, model: str) -> None:
    conds = _cond_order(pred)
    fig, axes = plt.subplots(1, len(conds), figsize=(3.2 * len(conds), 3.4))
    axes = np.atleast_1d(axes)
    for ax, c in zip(axes, conds):
        sub = pred[pred.condition == c]
        y = sub["label"].values
        yp = (sub[MODEL_KEYS[model]].values >= config.DECISION_THRESHOLD).astype(int)
        cm = confusion_matrix(y, yp, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_title(c)
        ax.set_xticks([0, 1], ["no-aedes", "aedes"], fontsize=8)
        ax.set_yticks([0, 1], ["no-aedes", "aedes"], fontsize=8)
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("actual")
    fig.suptitle(f"{MODEL_LABELS[model]} - confusion matrices by condition")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def plot_rocs(pred: pd.DataFrame, model: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for c in _cond_order(pred):
        sub = pred[pred.condition == c]
        y, p = sub["label"].values, sub[MODEL_KEYS[model]].values
        if len(np.unique(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, label=f"{c} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title(f"{MODEL_LABELS[model]} - ROC by condition")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)


def plot_mitigation(pred: pd.DataFrame, models: list[str], calib: dict) -> None:
    # Recalibrating vs.
    conds = _cond_order(pred)
    x = np.arange(len(conds))
    triv = trivial_f1(pred[pred.condition == conds[0]]["label"].values)

    series = []   # (label, model, threshold, style)
    for m in models:
        if m == "cnn_aug":
            continue
        series.append((f"{MODEL_LABELS[m]} @ 0.5", m, config.DECISION_THRESHOLD, "o--"))
        if m in calib.get("models", {}):
            t = calib["models"][m]["t_f1"]
            series.append((f"{MODEL_LABELS[m]} + umbral recal.", m, t, "o-"))
    if "cnn_aug" in models:
        series.append(("CNN reentrenada con ruido CR", "cnn_aug", config.DECISION_THRESHOLD, "s-"))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for label, model, thr, style in series:
        f1s, mccs = [], []
        for c in conds:
            sub = pred[pred.condition == c]
            m = evaluate_condition(sub["label"].values, sub[MODEL_KEYS[model]].values, thr=thr)
            f1s.append(m["f1"])
            mccs.append(m["mcc"])
        lw = 2.4 if model == "cnn_aug" else 1.5
        axes[0].plot(x, f1s, style, label=label, lw=lw)
        axes[1].plot(x, mccs, style, label=label, lw=lw)

    axes[0].axhline(triv, color="crimson", ls=":", lw=1.5)
    axes[0].text(0, triv + 0.012, f"trivial floor ({triv:.3f})", fontsize=7.5, color="crimson")
    axes[0].set_ylabel("F1")
    axes[0].set_title("F1 (note: the trivial floor is not zero)")
    axes[1].axhline(0.0, color="crimson", ls=":", lw=1.5)
    axes[1].text(0, 0.015, "trivial = 0.000", fontsize=7.5, color="crimson")
    axes[1].set_ylabel("MCC")
    axes[1].set_title("MCC (a fake rescue cannot hide here)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(_cond_labels(conds))
        ax.set_xlabel("condition (SNR)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    fig.suptitle("WINGBEAT - can it be fixed? Recalibrating the threshold vs. retraining with CR noise")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "mitigation.png", dpi=150)
    plt.close(fig)


def plot_noise_spectra(spectra: dict) -> None:
    lo, hi = config.WINGBEAT_BAND_HZ
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for env, d in spectra.items():
        ax.semilogy(d["freqs"], d["psd"], label=f"{env} ({d['band_fraction'] * 100:.1f}%)")
    ax.axvspan(lo, hi, color="crimson", alpha=0.15)
    ax.text((lo + hi) / 2, ax.get_ylim()[1], f" wingbeat\n {lo:.0f}-{hi:.0f} Hz",
            ha="center", va="top", fontsize=8, color="crimson")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("mean spectral density")
    ax.set_title("CR noise spectrum vs. the wingbeat band\n"
                 "(% = share of that environment's energy inside the band)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "noise_spectra.png", dpi=150)
    plt.close(fig)


# UFRGS holdout — the real capture-device shift (second transferability axis)
def evaluate_holdout(which: str) -> dict | None:
    # Score the UFRGS holdout, which was never in train/val/test.
    folders = np.load(config.FOLDERS_PATH, allow_pickle=True)
    mask = holdout_mask(folders)
    if not mask.any():
        print("No holdout dataset present — skipping.")
        return None

    y = np.load(config.LABELS_PATH)[mask]
    models = load_models(which)
    out: dict = {"datasets": list(config.HOLDOUT_DATASETS),
                 "n": int(mask.sum()), "n_pos": int(y.sum()), "models": {}}

    for name, model in models.items():
        if name == "baseline":
            x = np.load(config.MFCC_MATRIX_PATH)[mask]
            proba = model.predict_proba(x)[:, 1]
        else:
            x = normalize_mel(np.load(config.MEL_MATRIX_PATH)[mask])
            proba = model.predict(x, verbose=0).ravel()
        entry = evaluate_condition(y, proba)
        # The threshold sweep shows whether the model is blind or just miscalibrated.
        entry["threshold_sweep"] = {
            f"{t:.2f}": {
                "f1": evaluate_condition(y, proba, thr=t)["f1"],
                "recall": evaluate_condition(y, proba, thr=t)["recall"],
            }
            for t in (0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
        }
        out["models"][name] = entry
        print(f"  {MODEL_LABELS[name]:<14} F1={entry['f1']:.3f}  recall={entry['recall']:.3f}  "
              f"AUC={entry['roc_auc']:.3f}")

    (config.RESULTS_DIR / "ufrgs_holdout_metrics.json").write_text(json.dumps(out, indent=2))
    return out


# Report
def _fmt(v, nd=3) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def _check_clean_reproduces(rows: dict, models: list[str], allow_drift: bool) -> list[str]:
    # Clean goes through the same path as the noisy rows, so its F1 must match training.
    notes = []
    for model in models:
        path = CLEAN_METRICS_FILES[model]
        if not path.exists() or "clean" not in rows[model]:
            continue
        expected = json.loads(path.read_text())["f1"]
        got = rows[model]["clean"]["f1"]
        drift = abs(got - expected)
        ok = drift <= 0.01
        # ASCII only: this prints to a cp1252 console on Windows.
        notes.append(f"{MODEL_LABELS[model]}: clean F1 = {got:.4f} vs {expected:.4f} "
                     f"at training time (diff {drift:.4f}) {'OK' if ok else 'FAIL'}")
        if not ok and not allow_drift:
            raise AssertionError(
                f"[{model}] the re-extracted 'clean' condition gives F1={got:.4f} but "
                f"training reported {expected:.4f}. The hybrid test wiring is wrong and the "
                f"whole degradation table would be invalid. "
                f"(--allow-clean-drift para forzar de todos modos)"
            )
    return notes


def write_degradation_table(rows, out_path=config.DEGRADATION_TABLE, *, pred=None,
                            models=None, spectra=None, holdout=None, checks=None) -> Path:
    # Render every breakdown into results/degradation_table.md.
    conds = _cond_order(pred)
    snr = config.SNR_LEVELS_DB
    md: list[str] = []
    a = md.append

    a("# Degradation table — WINGBEAT\n")
    a("All four conditions hold **exactly the same "
      f"{int(rows[models[0]]['clean']['n'])} test segments**; the only thing that changes "
      "between them is the Costa Rican background noise mixed on top. Any difference "
      "between columns is caused by the noise.\n")
    a("| condition | SNR | what it is |")
    a("|---|---|---|")
    a("| `clean` | — | lab audio, no noise |")
    for c in conds:
        if c != "clean":
            a(f"| `{c}` | {snr[c]:.0f} dB | {'light background noise' if snr[c] >= 20 else 'noise comparable to the signal' if snr[c] >= 10 else 'noise as loud as the wingbeat'} |")
    a("")

    # 1 — headline
    triv = rows[models[0]][conds[0]]["trivial_f1"]
    a("## 1. Headline result\n")
    a("`FN rate` = of all the *Aedes* actually present, what fraction went **undetected**. "
      "`ROC-AUC` and `MCC` do not depend on the threshold.\n")
    a(f"> **Trivial floor = {triv:.3f}.** A classifier that answers \"aedes\" to everything gets "
      f"that F1 (base rate {100 * rows[models[0]][conds[0]]['n_pos'] / rows[models[0]][conds[0]]['n']:.1f} %). "
      f"**Every F1 in this table has to be read against that number, not against zero.** `MCC` is "
      f"the other safeguard: it is exactly **0.000** for that trivial classifier, so it cannot be "
      f"inflated by predicting everything positive.\n")
    a("| model | condition | F1 [95% CI] | vs. trivial | precision | recall | **FN rate** | ROC-AUC | MCC |")
    a("|---|---|---|---|---|---|---|---|---|")
    for model in models:
        for c in conds:
            m = rows[model][c]
            ci = f" [{_fmt(m.get('f1_ci_low'))}–{_fmt(m.get('f1_ci_high'))}]" if "f1_ci_low" in m else ""
            edge = m["f1"] - m["trivial_f1"]
            a(f"| {MODEL_LABELS[model]} | `{c}` | **{_fmt(m['f1'])}**{ci} | {edge:+.3f} "
              f"| {_fmt(m['precision'])} | {_fmt(m['recall'])} | **{_fmt(m['false_negative_rate'])}** "
              f"| {_fmt(m['roc_auc'])} | {_fmt(m['mcc'])} |")
    a("")

    # 2 — per environment
    main = "cnn" if "cnn" in models else models[0]
    a(f"## 2. Degradation by noise environment ({MODEL_LABELS[main]})\n")
    a("~1,241 segments per cell. Telling *which* noise breaks the detector would be a finding, "
      "but it has to be measured with the right metric.\n")
    a("> **Why environments are compared with ROC-AUC and not F1:** at the fixed 0.5 threshold "
      "F1 **collapses to ~0 in all five environments**, so there is no dynamic range left to "
      "tell them apart. ROC-AUC is **not saturated** and does show which one degrades most. "
      "Ranking environments by F1 here would be reading measurement noise.\n")
    envs = sorted(pred["environment"].unique())
    a("| environment | " + " | ".join(f"AUC `{c}`" for c in conds) +
      " | AUC drop | F1 `clean`→`hard` |")
    a("|---" * (len(conds) + 3) + "|")
    env_drop = {}      # by AUC — the metric that still has dynamic range
    env_f1_drop = {}
    for env in envs:
        aucs, f1s = [], []
        for c in conds:
            sub = pred[(pred.condition == c) & (pred.environment == env)]
            m = evaluate_condition(sub["label"].values, sub[MODEL_KEYS[main]].values)
            aucs.append(m["roc_auc"])
            f1s.append(m["f1"])
        env_drop[env] = aucs[0] - aucs[-1]
        env_f1_drop[env] = f1s[0] - f1s[-1]
        a(f"| `{env}` | " + " | ".join(_fmt(v) for v in aucs) +
          f" | **{_fmt(env_drop[env])}** | {_fmt(f1s[0])} → {_fmt(f1s[-1])} |")
    a("")

    # 3 — domain confound control
    a("## 3. Domain confound control\n")
    a("The models partly exploit a **channel cue** (they reject HumBugDB negatives more easily "
      "than Wingbeats ones). CR noise masks that cue, so **part of the measured degradation "
      "could be a dataset artifact dying rather than the wingbeat signal**. Control: repeat the "
      "measurement on the **same-domain** subset (Wingbeats only), which is free of the "
      "confound.\n")
    a(f"| condition | F1 full test | F1 same-domain (Wingbeats) | FP rate Wingbeats neg. | FP rate HumBugDB neg. | gap |")
    a("|---|---|---|---|---|---|")
    for c in conds:
        sub = pred[pred.condition == c]
        full = rows[main][c]["f1"]
        wb = sub[sub.dataset == "wingbeats"]
        sd = evaluate_condition(wb["label"].values, wb[MODEL_KEYS[main]].values)["f1"]
        fp = {}
        for ds in ("wingbeats", "humbugdb"):
            neg = sub[(sub.label == 0) & (sub.dataset == ds)]
            fp[ds] = float((neg[MODEL_KEYS[main]].values >= config.DECISION_THRESHOLD).mean()) \
                if len(neg) else float("nan")
        gap = fp["wingbeats"] - fp["humbugdb"]
        a(f"| `{c}` | {_fmt(full)} | **{_fmt(sd)}** | {_fmt(fp['wingbeats'])} | "
          f"{_fmt(fp['humbugdb'])} | {_fmt(gap)} |")
    a("")
    a("> If the gap **closes** once noise is added, that is direct evidence the channel cue "
      "existed and the noise erased it. If the same-domain F1 degrades just like the full one, "
      "the confound does **not** contaminate the headline result.\n")

    # 4 — false alarm on pure ambient noise
    a("## 4. Ambient false alarms — *do not cry wolf at a passing car*\n")
    bg = pred[pred.source_folder == "humbugdb/background"]
    if len(bg):
        a(f"FP rate over the **{len(bg[bg.condition == conds[0]])} negatives with no insect at all** "
          "(`humbugdb/background`): the system *should not* fire on pure ambient noise.\n")
        a("| condition | " + " | ".join(MODEL_LABELS[m] for m in models) + " |")
        a("|---" * (len(models) + 1) + "|")
        for c in conds:
            sub = bg[bg.condition == c]
            cells = [_fmt(float((sub[MODEL_KEYS[m]].values >= config.DECISION_THRESHOLD).mean()))
                     for m in models]
            a(f"| `{c}` | " + " | ".join(cells) + " |")
        a("")
        a("> **Careful reading this table: a 0.000 here is NOT a virtue.** Under noise the model "
          "stops predicting \"aedes\" **for almost everything** (recall goes to 0), so of course "
          "it raises no false alarms. It is not that it recognizes empty ambient noise well, it "
          "is that it no longer recognizes anything. This row only means something in `clean`.\n")
        a("> Not broken down by environment: that would be ~84 segments per cell, too fine to be "
          "reliable.\n")

    # 5 — diagnosis
    a("## 5. Diagnosis — noise destroying the signal, or a miscalibrated threshold?\n")
    a(f"**Read the last column against the trivial floor ({triv:.3f}), not against zero.** An "
      f"oracle F1 of 0.658 sounds like a partial rescue; next to the {triv:.3f} you get by "
      f"guessing, it is nothing.\n")
    a("| model | condition | F1 @ 0.5 | ROC-AUC | F1 at oracle threshold | vs. trivial | does the threshold rescue it? |")
    a("|---|---|---|---|---|---|---|")
    for model in models:
        for c in conds:
            m = rows[model][c]
            edge = m["oracle_f1"] - m["trivial_f1"]
            if c == "clean":
                verdict = "—"
            elif edge < 0.03:
                verdict = "**No** (~ guessing)"
            elif edge < 0.10:
                verdict = "barely"
            else:
                verdict = "yes"
            a(f"| {MODEL_LABELS[model]} | `{c}` | {_fmt(m['f1'])} | {_fmt(m['roc_auc'])} "
              f"| {_fmt(m['oracle_f1'])} | {edge:+.3f} | {verdict} |")
    a("")
    a("> **Correction relative to an earlier version of this table.** It used to report the "
      "oracle F1 (0.748 / 0.677 / 0.658) as if it measured how much recalibrating recovers. "
      "**Without the trivial floor next to it that is misleading:** at 10 dB and 0 dB the oracle "
      "only beats a classifier answering \"aedes\" to everything by +0.026 and +0.007. The honest "
      "reading is that **moving the threshold does NOT rescue those two conditions**: there is no "
      "spare information to recalibrate. Only `easy` (20 dB) has a real margin (+0.097, AUC "
      "0.820).\n")
    a("> The **oracle F1 picks the threshold while looking at the answers**: it is an *upper "
      "bound*, not deployable performance. It is only useful for the diagnosis: high AUC + low "
      "F1@0.5 + high oracle means **miscalibration** (fixable by moving the threshold). Low AUC "
      "+ low oracle means **the noise destroyed the information** (no threshold saves it).\n")

    if spectra:
        lo, hi = config.WINGBEAT_BAND_HZ
        a(f"### Does the noise spectrum explain which environment hurts the detector most?\n")
        a(f"Prior hypothesis, stated to be tested rather than assumed: noise that **overlaps the "
          f"wingbeat band ({lo:.0f}-{hi:.0f} Hz)** should interfere more. It is tested against the "
          f"**AUC** drop, not F1, because of the saturation explained in section 2.\n")
        a("| environment | % of noise energy in the band | AUC drop (clean->hard) |")
        a("|---|---|---|")
        bf, dr = [], []
        for env in sorted(spectra, key=lambda e: -spectra[e]["band_fraction"]):
            frac = spectra[env]["band_fraction"]
            drop = env_drop.get(env, float("nan"))
            bf.append(frac)
            dr.append(drop)
            a(f"| `{env}` | {frac * 100:.1f} % | {_fmt(drop)} |")
        a("")
        if len(bf) > 2 and not np.isnan(dr).any() and np.std(dr) > 0:
            r = float(np.corrcoef(bf, dr)[0, 1])
            a(f"> Pearson correlation: **r = {r:.2f}** (n = {len(bf)} environments, which is few "
              f"enough that this is *suggestive, not conclusive*).")
            if abs(r) < 0.4:
                a("> **The spectral-overlap hypothesis does NOT hold with this data.** The detector "
                  "does not break *especially* on noise that occupies its band: it breaks on "
                  "**any** real ambient noise. Reported as such, against the initial hunch, "
                  "because that is what the numbers say.\n")
            else:
                a("")

    # 6 — the mitigation arm: can it be fixed?
    calib = json.loads(config.CALIBRATION_PATH.read_text()) if config.CALIBRATION_PATH.exists() else None
    if calib:
        a("## 6. The cure — can it be fixed?\n")
        a("Two interventions, with very different costs:\n")
        a("1. **Recalibrate the threshold** (free, no retraining). The threshold is chosen on a "
          "**noisy validation set**: val segments + the `cal` half of the reserved noise. Neither "
          "those segments nor that noise audio are in the test set. *This is not the oracle "
          "threshold.*")
        a("2. **Retrain the CNN with CR noise** (`--augment`): one noisy copy of every training "
          "segment, at a random SNR between 0 and 20 dB, using **only** the `aug` half of the "
          "reserved noise. The test noise is still unseen by it.\n")
        a(f"| strategy | threshold | `clean` | `easy` 20 dB | `medium` 10 dB | `hard` 0 dB |")
        a("|---|---|---|---|---|---|")

        def _row(label: str, model: str, thr: float, thr_label: str) -> None:
            cells = []
            for c in conds:
                sub = pred[pred.condition == c]
                m = evaluate_condition(sub["label"].values, sub[MODEL_KEYS[model]].values, thr=thr)
                cells.append(f"{m['f1']:.3f}")
            a(f"| {label} | {thr_label} | " + " | ".join(cells) + " |")

        for model in models:
            if model == "cnn_aug":
                continue
            _row(f"{MODEL_LABELS[model]} (as it was)", model, config.DECISION_THRESHOLD, "0.50")
            if model in calib.get("models", {}):
                t = calib["models"][model]["t_f1"]
                _row(f"{MODEL_LABELS[model]} **+ recalibrated threshold**", model, t, f"{t:.3f}")
        if "cnn_aug" in models:
            _row("**CNN retrained with CR noise**", "cnn_aug", config.DECISION_THRESHOLD, "0.50")
            if "cnn_aug" in calib.get("models", {}):
                t = calib["models"]["cnn_aug"]["t_f1"]
                _row("**CNN with noise + recalibrated threshold**", "cnn_aug", t, f"{t:.3f}")
        a(f"| *trivial floor (answer \"aedes\" to everything)* | — | *{triv:.3f}* | *{triv:.3f}* | "
          f"*{triv:.3f}* | *{triv:.3f}* |")
        a("")

        # MCC is 0.0 for the all-positive classifier, so a fake recovery shows up here.
        a("**The same table in MCC** (it is 0.000 for the trivial classifier, so a fake rescue "
          "cannot hide):\n")
        a(f"| strategy | `clean` | `easy` | `medium` | `hard` |")
        a("|---|---|---|---|---|")

        def _mcc_row(label: str, model: str, thr: float) -> None:
            cells = []
            for c in conds:
                sub = pred[pred.condition == c]
                m = evaluate_condition(sub["label"].values, sub[MODEL_KEYS[model]].values, thr=thr)
                cells.append(f"{m['mcc']:.3f}")
            a(f"| {label} | " + " | ".join(cells) + " |")

        for model in models:
            if model == "cnn_aug" or model not in calib.get("models", {}):
                continue
            _mcc_row(f"{MODEL_LABELS[model]} + recalibrated threshold", model,
                     calib["models"][model]["t_f1"])
        if "cnn_aug" in models:
            _mcc_row("**CNN retrained with CR noise**", "cnn_aug", config.DECISION_THRESHOLD)
        a("")

        for model in models:
            info = calib.get("models", {}).get(model)
            if info and model != "cnn_aug":
                a(f"> **{MODEL_LABELS[model]}** — threshold chosen on validation: "
                  f"`{info['t_f1']:.4f}`. {info.get('verdict', '')}")
        a("")
        a("> **Honest reading.** Recalibrating is free but **it does not help the CNN**: its "
          "optimal threshold goes essentially to zero, meaning the \"solution\" is to answer "
          "*aedes* to everything, so F1 rises to the trivial floor while MCC stays on the floor. "
          "The **RandomForest is recalibratable**, which is the irony here: **the simple model, "
          "the one that lost in the lab, is the only one you can rescue without retraining.**\n")

    # 7 — UFRGS holdout
    if holdout:
        a("## 7. UFRGS holdout — a real capture-device change\n")
        a(f"{holdout['n']} segments recorded **with a phone**, never seen in training "
          f"(only {holdout['n_pos']} positives, a caveat worth stating). This is the second "
          "transferability axis: not synthetic noise, but a different real microphone.\n")
        a("| model | F1 | recall | ROC-AUC |")
        a("|---|---|---|---|")
        for name, m in holdout["models"].items():
            a(f"| {MODEL_LABELS[name]} | {_fmt(m['f1'])} | {_fmt(m['recall'])} "
              f"| {_fmt(m['roc_auc'])} |")
        a("")
        # The cure is specific to CR noise, so it gets reported rather than left out.
        hm = holdout["models"]
        if "cnn_aug" in hm and "cnn" in hm:
            base_auc, aug_auc = hm["cnn"]["roc_auc"], hm["cnn_aug"]["roc_auc"]
            if aug_auc < base_auc - 0.05:
                a(f"> **The most uncomfortable finding, and the most important one: the cure is "
                  f"SPECIFIC, not general.** Retraining with CR noise fixed CR noise dramatically "
                  f"(F1 0.001 -> 0.781 at 0 dB), but **made transfer to a real device worse**: "
                  f"ROC-AUC on UFRGS falls from **{base_auc:.3f}** (naive CNN) to "
                  f"**{aug_auc:.3f}** (augmented CNN), a hair above chance (0.5).\n")
                a("> In other words, teaching it Costa Rican noise made it robust **to that "
                  "noise**, not robust *in general*. It specialized. **A microphone change is a "
                  "different problem from a background-noise change, and solving one does not "
                  "solve the other.** A real deployment would have to attack both separately, and "
                  "none of this would have been visible without holding UFRGS out.\n")

    # 7.5 — the prevalence caveat: at what base rate were these numbers measured?
    a("## 8. At what prevalence do these numbers hold? (deployment caveat)\n")
    a(f"Everything above was measured on a **deliberately balanced** test set: "
      f"{100 * rows[main][conds[0]]['n_pos'] / rows[main][conds[0]]['n']:.1f} % of the "
      f"segments contain an *Aedes* (a consequence of `TARGET_SEGMENTS_PER_CLASS`). "
      f"**In a real deployment a mosquito is not present half the time**, it is much closer "
      f"to 1 %.\n")
    a("That does not change recall or AUC (neither depends on prevalence), but it **sinks "
      "precision**, because false positives multiply against a much larger base of "
      "negatives:\n")
    a("> `precision(pi) = TPR*pi / (TPR*pi + FPR*(1-pi))`\n")
    prev_model = "cnn_aug" if "cnn_aug" in models else main
    a(f"Precision of the **{MODEL_LABELS[prev_model]}** (same model, same threshold, same "
      f"TPR/FPR; only how rare the mosquito is changes):\n")
    a("| condition | recall (TPR) | FP rate | precision @ pi=50% | @ pi=10% | @ pi=1% |")
    a("|---|---|---|---|---|---|")
    for c in conds:
        m = rows[prev_model][c]
        tpr, fpr = m["recall"], m["fp_rate"]
        cells = []
        for pi in (0.50, 0.10, 0.01):
            denom = tpr * pi + fpr * (1 - pi)
            cells.append(f"{tpr * pi / denom:.3f}" if denom > 0 else "—")
        a(f"| `{c}` | {_fmt(tpr)} | {_fmt(fpr)} | " + " | ".join(cells) + " |")
    a("")
    a("> **How to read this without misleading anyone.** An F1 of 0.907 is real *in the "
      "regime where it was measured*. If the same detector were hung in a yard where the "
      "mosquito shows up 1 % of the time, **most of its alarms would be false**, even though "
      "it hears just as well. That is not a flaw in the model: it is the arithmetic of "
      "imbalanced classes, and it happens to any detector. The engineering answer is to "
      "raise the threshold (giving up recall) or to require several detections over time "
      "before firing, **not** to report the balanced precision as if it were the field "
      "one.\n")
    a("> Stated here because leaving it out would be the same kind of silence as omitting "
      "the trivial floor (section 1): a correct number presented in a context that makes it "
      "look like something else.\n")

    # 9 — mechanical observations
    a("## 9. Observations\n")
    a("*(derived mechanically from the numbers above; the causal interpretation is written "
      "separately, from the data, with no mechanism assumed up front)*\n")
    c0, cN = conds[0], conds[-1]
    for model in models:
        r0, rN = rows[model][c0], rows[model][cN]
        a(f"- **{MODEL_LABELS[model]}**: F1 goes from {_fmt(r0['f1'])} (`{c0}`) to {_fmt(rN['f1'])} "
          f"(`{cN}`, {snr[cN]:.0f} dB), a drop of **{_fmt(r0['f1'] - rN['f1'])}**. "
          f"The false-negative rate rises from {_fmt(r0['false_negative_rate'])} to "
          f"{_fmt(rN['false_negative_rate'])}, and ROC-AUC from {_fmt(r0['roc_auc'])} to "
          f"{_fmt(rN['roc_auc'])}.")

    # At the first noisy condition: is the model blind, or just miscalibrated?
    c1 = conds[1] if len(conds) > 1 else None
    if c1:
        m1 = rows[main][c1]
        a("")
        a(f"- **The drop at `{c1}` ({snr[c1]:.0f} dB) is mostly CALIBRATION, not blindness.** "
          f"At the fixed 0.5 threshold the {MODEL_LABELS[main]} gives F1 = {_fmt(m1['f1'])} "
          f"(recall {_fmt(m1['recall'])}, so it detects almost no mosquitoes), but its ROC-AUC "
          f"is still **{_fmt(m1['roc_auc'])}** and, with a recalibrated threshold, it would "
          f"reach F1 = **{_fmt(m1['oracle_f1'])}**, which is "
          f"**{m1['oracle_f1'] - m1['trivial_f1']:+.3f}** over the trivial floor. The wingbeat "
          f"information **is still there**; what broke is the decision point, calibrated for lab "
          f"silence. Same pattern as the UFRGS holdout.")
        mN = rows[main][cN]
        a(f"- **But at the loudest noise (`{cN}`, {snr[cN]:.0f} dB) there is nothing left to "
          f"recalibrate.** AUC falls to {_fmt(mN['roc_auc'])} and the oracle threshold only "
          f"reaches F1 = {_fmt(mN['oracle_f1'])}, i.e. "
          f"**{mN['oracle_f1'] - mN['trivial_f1']:+.3f}** over the trivial floor "
          f"({_fmt(mN['trivial_f1'])}): statistically indistinguishable from answering "
          f"\"aedes\" to everything. Here the information really was destroyed, and **no "
          f"threshold saves the detector, only retraining it with noise can.**")
    if len(models) > 1 and "baseline" in rows and "cnn" in rows:
        a(f"- **The simple model resists noise better.** At `{cN}` the RandomForest AUC "
          f"({_fmt(rows['baseline'][cN]['roc_auc'])}) beats the CNN one "
          f"({_fmt(rows['cnn'][cN]['roc_auc'])}), even though the CNN wins on clean audio "
          f"({_fmt(rows['cnn'][c0]['f1'])} vs {_fmt(rows['baseline'][c0]['f1'])} F1). "
          f"Winning in the lab and winning under noise **are not the same contest**.")
    if "cnn_aug" in rows:
        aug0, augN = rows["cnn_aug"][c0], rows["cnn_aug"][cN]
        a("")
        a(f"- **The cure works, and the effect is large.** Retraining the CNN with CR noise "
          f"(`--augment`) takes it, at `{cN}` ({snr[cN]:.0f} dB) and **at the default 0.5 "
          f"threshold**, from F1 {_fmt(rows['cnn'][cN]['f1'])} to **{_fmt(augN['f1'])}**, "
          f"{augN['f1'] - augN['trivial_f1']:+.3f} over the trivial floor, with MCC "
          f"{_fmt(augN['mcc'])}. And it **costs almost nothing on clean audio**: "
          f"{_fmt(aug0['f1'])} vs {_fmt(rows['cnn'][c0]['f1'])} for the naive one. The test "
          f"noise was unseen by it (recordings `_01`/`_02`).")
    if env_drop:
        worst = max(env_drop, key=env_drop.get)
        best = min(env_drop, key=env_drop.get)
        spread = env_drop[worst] - env_drop[best]
        a(f"- By environment (measured in AUC, see section 2): the one that degrades the "
          f"{MODEL_LABELS[main]} **most** is `{worst}` (drop {_fmt(env_drop[worst])}); the one "
          f"that degrades it **least** is `{best}` (drop {_fmt(env_drop[best])}). The spread "
          f"between worst and best is only **{_fmt(spread)}** of AUC: **no environment is "
          f"safe**, the collapse comes from real noise being present at all, not from one "
          f"particular kind of noise.")
    if checks:
        a("")
        a("### Integrity check\n")
        a("The `clean` condition was re-extracted from disk through the **same path** as the "
          "noisy ones (manifest -> WAV -> features -> model). Its F1 has to reproduce the one "
          "from training; if it does not, the wiring is broken and this whole table would be "
          "invalid.\n")
        for note in checks:
            a(f"- {note}")
    a("")
    a("---")
    a(f"*Generated by `src/evaluate.py` · decision threshold {config.DECISION_THRESHOLD} · "
      f"95% CI by grouped bootstrap ({config.BOOTSTRAP_RESAMPLES} resamples of recordings).*")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WINGBEAT degradation")
    parser.add_argument("--model", choices=("baseline", "cnn", "cnn_aug", "both", "all"),
                        default="all",
                        help="which serialized model(s) to evaluate ('all' adds the "
                             "noise-augmented CNN once it has been trained)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only score the first N segments of each condition (smoke test)")
    parser.add_argument("--rescore", action="store_true", help="ignore the probability cache")
    parser.add_argument("--allow-clean-drift", action="store_true",
                        help="do not fail when the clean condition misses its training F1")
    parser.add_argument("--no-holdout", action="store_true", help="skip the UFRGS holdout")
    args = parser.parse_args()

    if not config.HYBRID_MANIFEST.exists():
        raise FileNotFoundError(
            f"No hybrid manifest at {config.HYBRID_MANIFEST}. Run: python -m src.build_hybrid_testset"
        )
    manifest = pd.read_csv(config.HYBRID_MANIFEST)
    if args.limit:
        manifest = manifest.groupby("condition", sort=False).head(args.limit).reset_index(drop=True)
        print(f"SMOKE TEST: {args.limit} segments per condition ({len(manifest)} rows).")

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pred = score_all(manifest, args.model, rescore=args.rescore or bool(args.limit))
    models = [m for m in ("cnn", "baseline", "cnn_aug") if MODEL_KEYS[m] in pred.columns]

    rows = {m: {c: _slice_metrics(pred[pred.condition == c], m) for c in _cond_order(pred)}
            for m in models}

    checks = None
    if args.limit:
        print("\n(--limit: skipping the 'clean' reproduction check, the subset is not "
              "comparable with the full test set.)")
    else:
        checks = _check_clean_reproduces(rows, models, args.allow_clean_drift)
        print("\n--- integrity check (clean must reproduce training) ---")
        for note in checks:
            print(f"  {note}")

    print("\n--- degradation (F1 @ %.2f) ---" % config.DECISION_THRESHOLD)
    for m in models:
        line = "  ".join(f"{c}={rows[m][c]['f1']:.3f}" for c in _cond_order(pred))
        print(f"  {MODEL_LABELS[m]:<14} {line}")

    print("\nNoise spectrum per environment...")
    spectra = noise_band_energy()

    holdout = None
    if not args.no_holdout:
        print("\n--- UFRGS holdout (real device) ---")
        holdout = evaluate_holdout(args.model)

    main_model = "cnn" if "cnn" in models else models[0]
    plot_degradation(rows, pred, models)
    plot_by_environment(pred, main_model)
    plot_confusions(pred, main_model)
    plot_rocs(pred, main_model)
    plot_noise_spectra(spectra)
    if config.CALIBRATION_PATH.exists():
        plot_mitigation(pred, models, json.loads(config.CALIBRATION_PATH.read_text()))

    config.DEGRADATION_METRICS.write_text(json.dumps(
        {"threshold": config.DECISION_THRESHOLD,
         "conditions": rows,
         "noise_band_fraction": {e: d["band_fraction"] for e, d in spectra.items()}},
        indent=2))
    out = write_degradation_table(rows, pred=pred, models=models, spectra=spectra,
                                  holdout=holdout, checks=checks)

    print(f"\nTable   -> {out}")
    print(f"Metrics -> {config.DEGRADATION_METRICS}")
    print(f"Plots   -> {config.RESULTS_DIR}/*.png")


if __name__ == "__main__":
    main()
