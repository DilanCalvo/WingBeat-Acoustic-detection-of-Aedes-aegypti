from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from audiomentations import AddBackgroundNoise
from tqdm import tqdm

from . import config
from .data_split import grouped_split
from .preprocess import normalize_volume

MANIFEST_COLUMNS = [
    "condition", "snr_db", "environment", "label", "class_name", "dataset",
    "source_folder", "source_recording", "source_segment_path", "hybrid_path",
]


def list_cr_noise_files() -> dict[str, dict[str, list[Path]]]:
    # Group the CR noise recordings by environment, split into test/train.
    hold = config.NOISE_FILES_HELD_FOR_TRAIN
    pools: dict[str, dict[str, list[Path]]] = {}
    for env in config.CR_NOISE_ENVIRONMENTS:
        files = sorted(config.NOISE_CR_DIR.glob(f"{env}*.wav"))
        if not files:
            raise FileNotFoundError(
                f"No CR noise recordings for '{env}' in {config.NOISE_CR_DIR}. "
                f"Expected files named '{env}_*.wav'."
            )
        if hold and len(files) <= hold:
            raise ValueError(
                f"'{env}' has {len(files)} recording(s) but {hold} is reserved for "
                f"train; none left for the test set. Record more or lower "
                f"config.NOISE_FILES_HELD_FOR_TRAIN."
            )
        train = files[len(files) - hold:] if hold else []
        test = files[:len(files) - hold] if hold else files
        pools[env] = {"test": test, "train": train}
    return pools


def assign_environments(labels) -> list[str]:
    # One environment per segment, round-robin within each class.
    envs = list(config.CR_NOISE_ENVIRONMENTS)
    counters: dict[int, int] = {}
    out: list[str] = []
    for lab in labels:
        k = counters.get(int(lab), 0)
        out.append(envs[k % len(envs)])
        counters[int(lab)] = k + 1
    return out


def build_noise_transforms(pools, snr_levels: dict[str, float]):
    # One AddBackgroundNoise per (environment, condition).
    transforms = {}
    for env in config.CR_NOISE_ENVIRONMENTS:
        test_files = [str(p) for p in pools[env]["test"]]
        for cond, snr in snr_levels.items():
            transforms[(env, cond)] = AddBackgroundNoise(
                sounds_path=test_files,
                min_snr_db=snr,
                max_snr_db=snr,
                noise_rms="relative",
                p=1.0,
            )
    return transforms


def make_noisy_copy(waveform: np.ndarray, transform: AddBackgroundNoise) -> np.ndarray:
    # Mix noise onto one segment, then peak-normalize.
    noisy = transform(samples=waveform.astype(np.float32), sample_rate=config.SAMPLE_RATE)
    return normalize_volume(noisy, target_peak=config.NOISE_MIX_TARGET_PEAK)


def _hybrid_path(condition: str, segment_path: str, root: Path = config.HYBRID_DIR) -> Path:
    # Destination <root>/<condition>/<class>/<file>.wav for a segment.
    rel = (config.PROJECT_ROOT / segment_path).relative_to(config.INTERIM_DIR)
    return root / condition / rel


def _reset_condition_dirs(conditions, root: Path = config.HYBRID_DIR) -> None:
    # Clear the condition folders so a re-run never mixes with stale files.
    for cond in conditions:
        d = root / cond
        if d.exists():
            shutil.rmtree(d)


def build_all_conditions(snr_levels: dict[str, float], limit: int | None = None,
                         split: str = "test") -> Path:
    # Generate clean + easy/medium/hard for one split and write the manifest.
    manifest = pd.read_csv(config.SEGMENTS_MANIFEST)
    groups = np.load(config.GROUPS_PATH)
    folders = np.load(config.FOLDERS_PATH, allow_pickle=True)

    # The feature arrays must be row-aligned with the manifest.
    assert len(manifest) == len(groups) == len(folders), (
        "processed arrays are stale vs the manifest — re-run src.features"
    )
    assert (manifest["source_folder"].values == folders).all(), (
        "manifest/folders.npy misaligned — re-run src.features"
    )

    _, val_idx, test_idx = grouped_split(groups, folders)   # excludes the UFRGS holdout
    idx = test_idx if split == "test" else val_idx
    root = config.HYBRID_DIR if split == "test" else config.HYBRID_VAL_DIR
    out_manifest = config.HYBRID_MANIFEST if split == "test" else config.HYBRID_VAL_MANIFEST

    test = manifest.iloc[idx].reset_index(drop=True)
    if limit:
        test = test.iloc[:limit].reset_index(drop=True)

    envs = assign_environments(test["label"].tolist())
    if split == "test":
        pools = list_cr_noise_files()
    else:
        # Lazy import: only the val path needs the calibration-half noise.
        from .build_noise_augmentation import prepare_noise_pools
        cal = prepare_noise_pools()["cal"]
        pools = {env: {"test": cal[env], "train": []} for env in config.CR_NOISE_ENVIRONMENTS}
    transforms = build_noise_transforms(pools, snr_levels)

    _reset_condition_dirs(snr_levels.keys(), root)
    root.mkdir(parents=True, exist_ok=True)
    random.seed(config.SEED)   # audiomentations draws from the global RNG

    rows: list[dict] = []

    def meta(r) -> dict:
        return {
            "label": int(r.label), "class_name": r.class_name, "dataset": r.dataset,
            "source_folder": r.source_folder, "source_recording": r.source_recording,
            "source_segment_path": r.segment_path,
        }

    # Clean: reference the original interim WAV, no copy.
    for i, r in enumerate(test.itertuples(index=False)):
        rows.append({"condition": "clean", "snr_db": "", "environment": envs[i],
                     **meta(r), "hybrid_path": r.segment_path})

    # easy / medium / hard: write a noised copy of every segment.
    for cond, snr in snr_levels.items():
        for i, r in enumerate(tqdm(test.itertuples(index=False), total=len(test), desc=f"{cond:<7}")):
            wav, _ = sf.read(str(config.PROJECT_ROOT / r.segment_path))
            noisy = make_noisy_copy(wav, transforms[(envs[i], cond)])
            out = _hybrid_path(cond, r.segment_path, root)
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, noisy, config.SAMPLE_RATE, subtype="PCM_16")
            rows.append({"condition": cond, "snr_db": snr, "environment": envs[i],
                         **meta(r),
                         "hybrid_path": out.relative_to(config.PROJECT_ROOT).as_posix()})

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df.to_csv(out_manifest, index=False)

    print(f"\nDone [{split}]. {len(df)} rows across {df['condition'].nunique()} conditions "
          f"({len(test)} {split} segments x {1 + len(snr_levels)} conditions).")
    print("\nSegments per condition:")
    print(df.groupby("condition").size().reindex(["clean", *snr_levels]).to_string())
    print("\nEnvironment balance (noisy rows, positives / total):")
    noisy = df[df.condition != "clean"]
    bal = noisy.groupby("environment").agg(total=("label", "size"),
                                           positives=("label", "sum"))
    print(bal.to_string())
    print(f"\nManifest: {out_manifest}")
    return out_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the hybrid SNR test set")
    parser.add_argument(
        "--snr", nargs="*", metavar="NAME=DB", default=None,
        help="override SNR levels, e.g. --snr easy=25 hard=-5 (default: config.SNR_LEVELS_DB)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="only process the first N test segments (smoke test)",
    )
    parser.add_argument(
        "--split", choices=("test", "val"), default="test",
        help="test = the central experiment (test noise pool); "
             "val = the noisy validation set for threshold calibration (calibration-half noise)",
    )
    args = parser.parse_args()

    snr_levels = dict(config.SNR_LEVELS_DB)
    if args.snr:
        for tok in args.snr:
            name, _, val = tok.partition("=")
            if not val:
                parser.error(f"--snr expects NAME=DB tokens, got '{tok}'")
            snr_levels[name] = float(val)

    build_all_conditions(snr_levels, args.limit, args.split)


if __name__ == "__main__":
    main()
