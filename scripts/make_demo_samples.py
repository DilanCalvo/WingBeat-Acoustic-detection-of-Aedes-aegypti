from __future__ import annotations

import argparse
import shutil

import numpy as np
import pandas as pd

from src import config, inference
from src.data_split import grouped_split

DEMO_SAMPLES_DIR = config.PROJECT_ROOT / "demo" / "samples"
PREFERRED_NEG_FOLDERS = (
    "humbugdb/background",
    "wingbeats/Ae. albopictus",
    "wingbeats/C. pipiens",
    "humbugdb/an_arabiensis",
    "wingbeats/An. gambiae",
    "wingbeats/C. quinquefasciatus",
)


def _slug(folder: str) -> str:
    tail = folder.split("/")[-1].replace(".", " ").replace("_", " ").split()
    return tail[-1].lower() if tail else "clip"


def _test_frame() -> pd.DataFrame:
    manifest = pd.read_csv(config.SEGMENTS_MANIFEST)
    groups = np.load(config.GROUPS_PATH)
    folders = np.load(config.FOLDERS_PATH, allow_pickle=True)
    assert len(manifest) == len(groups) == len(folders), (
        "processed arrays are stale vs the manifest — re-run src.features"
    )
    _, _, test_idx = grouped_split(groups, folders)
    return manifest.iloc[test_idx].reset_index(drop=True)


def _score(row, models) -> dict:
    wav = inference.prepare_waveform(config.PROJECT_ROOT / row.segment_path)
    if inference.count_segments(wav) == 0:
        return {}
    noisy = inference.add_cr_noise(wav, "street_traffic", config.SNR_LEVELS_DB["hard"])
    clean = inference.predict_all(wav, models)
    hard = inference.predict_all(noisy, models)
    return {
        "cnn_clean": clean["cnn"].aggregate,
        "cnn_hard": hard["cnn"].aggregate,
        "aug_hard": hard["cnn_aug"].aggregate,
        "cnn_neg_clean": clean["cnn"].aggregate,  # alias for readability in negatives
    }


def select_samples(n_pos: int, n_neg: int, candidates: int, models) -> pd.DataFrame:
    test = _test_frame()
    rng = np.random.default_rng(config.SEED)

    pos = test[test.label == config.LABEL_AEDES]
    pos_cand = pos.sample(min(candidates, len(pos)), random_state=config.SEED)
    pos_rows = []
    for r in pos_cand.itertuples():
        s = _score(r, models)
        if not s:
            continue
        score = s["cnn_clean"] - s["cnn_hard"] + s["aug_hard"]
        pos_rows.append({"segment_path": r.segment_path, "label": 1,
                         "source_folder": r.source_folder,
                         "source_recording": r.source_recording, "narr_score": score, **s})
    pos_df = pd.DataFrame(pos_rows).sort_values("narr_score", ascending=False)
    chosen_pos = _pick_distinct(pos_df, n_pos)

    #negatives: clearest rejections, one per preferred folder for diversity
    neg = test[test.label == config.LABEL_NON_AEDES]
    present = [f for f in PREFERRED_NEG_FOLDERS if f in set(neg.source_folder)]
    present += [f for f in sorted(set(neg.source_folder)) if f not in present]
    neg_rows = []
    for folder in present:
        pool = neg[neg.source_folder == folder]
        for r in pool.sample(min(6, len(pool)), random_state=config.SEED).itertuples():
            s = _score(r, models)
            if not s:
                continue
            neg_rows.append({"segment_path": r.segment_path, "label": 0,
                             "source_folder": folder,
                             "source_recording": r.source_recording,
                             "cnn_clean": s["cnn_clean"]})
    neg_df = pd.DataFrame(neg_rows)
    chosen_neg = _pick_negatives(neg_df, present, n_neg)

    return pd.concat([chosen_pos, chosen_neg], ignore_index=True)


def _pick_distinct(df: pd.DataFrame, n: int) -> pd.DataFrame:
    #Top-n rows, no two from the same source_recording (diversity)
    picked, seen = [], set()
    for _, row in df.iterrows():
        if row.source_recording in seen:
            continue
        picked.append(row)
        seen.add(row.source_recording)
        if len(picked) == n:
            break
    return pd.DataFrame(picked)


def _pick_negatives(df: pd.DataFrame, folder_order, n: int) -> pd.DataFrame:
    #One clearest (lowest cnn_clean) negative per folder, in priority order, up to n
    picked = []
    for folder in folder_order:
        pool = df[df.source_folder == folder].sort_values("cnn_clean")
        if len(pool):
            picked.append(pool.iloc[0])
        if len(picked) == n:
            break
    return pd.DataFrame(picked)


def _write(chosen: pd.DataFrame) -> None:
    #Copy chosen interim WAVs into demo/samples/ with readable {pos|neg}_<slug>_<i> names
    DEMO_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for old in DEMO_SAMPLES_DIR.glob("*.wav"):  # clean re-run
        old.unlink()

    counters: dict[str, int] = {}
    for row in chosen.itertuples():
        prefix = "pos" if row.label == 1 else "neg"
        slug = _slug(row.source_folder)
        key = f"{prefix}_{slug}"
        counters[key] = counters.get(key, 0) + 1
        dest = DEMO_SAMPLES_DIR / f"{key}_{counters[key]:02d}.wav"
        shutil.copy(config.PROJECT_ROOT / row.segment_path, dest)
        print(f"  {dest.name:24s} <- {row.segment_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate committed demo sample clips")
    parser.add_argument("--n-pos", type=int, default=4)
    parser.add_argument("--n-neg", type=int, default=4)
    parser.add_argument("--candidates", type=int, default=80,
                        help="positive candidates to score before ranking")
    args = parser.parse_args()

    models = inference.load_all_models()
    missing = {"cnn", "cnn_aug", "baseline"} - set(models)
    if missing:
        raise SystemExit(f"Missing models {missing} — train them before curating demo samples.")

    chosen = select_samples(args.n_pos, args.n_neg, args.candidates, models)
    print("\nChosen samples (narrative verified by scoring):")
    cols = ["source_folder", "cnn_clean", "cnn_hard", "aug_hard"]
    print(chosen[[c for c in cols if c in chosen]].to_string())
    print(f"\nWriting {len(chosen)} clips to {DEMO_SAMPLES_DIR.relative_to(config.PROJECT_ROOT)} ...")
    _write(chosen)
    print("\nDone. Commit demo/samples/ so the demo runs offline.")


if __name__ == "__main__":
    main()
