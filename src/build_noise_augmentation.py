# Splits each reserved *_03 recording in time into an `aug` half (mixed into training.

from __future__ import annotations

import argparse
import random
import shutil

import numpy as np
import pandas as pd
import soundfile as sf
from audiomentations import AddBackgroundNoise
from tqdm import tqdm

from . import config
from .build_hybrid_testset import list_cr_noise_files
from .data_split import grouped_split
from .preprocess import normalize_volume


def prepare_noise_pools() -> dict[str, dict[str, list]]:
    # Split each reserved *_03 recording into disjoint `aug` and `cal` halves.
    pools = list_cr_noise_files()
    out: dict[str, dict[str, list]] = {"aug": {}, "cal": {}}

    for half in ("aug", "cal"):
        (config.NOISE_POOL_DIR / half).mkdir(parents=True, exist_ok=True)

    for env, pool in pools.items():
        out["aug"][env], out["cal"][env] = [], []
        for src in pool["train"]:          # the RESERVED recordings only
            wav, sr = sf.read(str(src))
            cut = int(len(wav) * config.NOISE_AUG_SPLIT)
            for half, chunk in (("aug", wav[:cut]), ("cal", wav[cut:])):
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                if rms < config.MIN_SEGMENT_RMS:
                    raise ValueError(
                        f"The '{half}' half of {src.name} is silent (RMS {rms:.2e}). "
                        f"Augmenting with silence would do nothing."
                    )
                dst = config.NOISE_POOL_DIR / half / src.name
                sf.write(dst, chunk, sr, subtype="PCM_16")
                out[half][env].append(dst)
    return out


def build_augmented_train(limit: int | None = None) -> pd.DataFrame:
    # One noisy copy of every train segment, at a random SNR.
    manifest = pd.read_csv(config.SEGMENTS_MANIFEST)
    groups = np.load(config.GROUPS_PATH)
    folders = np.load(config.FOLDERS_PATH, allow_pickle=True)
    assert len(manifest) == len(groups) == len(folders), "stale arrays — re-run src.features"

    train_idx, val_idx, test_idx = grouped_split(groups, folders)
    train = manifest.iloc[train_idx].reset_index(drop=True)
    if limit:
        train = train.iloc[:limit].reset_index(drop=True)

    # The augmented copies must not bring in a recording val or test also holds.
    aug_groups = set(groups[train_idx].tolist())
    assert aug_groups.isdisjoint(set(groups[val_idx].tolist())), "aug leaks into val"
    assert aug_groups.isdisjoint(set(groups[test_idx].tolist())), "aug leaks into test"

    aug_pool = prepare_noise_pools()["aug"]
    lo, hi = config.AUG_SNR_RANGE_DB
    envs = list(config.CR_NOISE_ENVIRONMENTS)

    # One mixer per environment; audiomentations draws the SNR uniformly per call.
    transforms = {
        env: AddBackgroundNoise(
            sounds_path=[str(p) for p in aug_pool[env]],
            min_snr_db=lo, max_snr_db=hi, noise_rms="relative", p=1.0,
        )
        for env in envs
    }

    if config.AUG_DIR.exists():
        shutil.rmtree(config.AUG_DIR)
    config.AUG_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(config.SEED)      # audiomentations draws from the global RNG

    rows = []
    for i, r in enumerate(tqdm(train.itertuples(index=False), total=len(train), desc="augment")):
        env = envs[i % len(envs)]             # round-robin: every environment equally represented
        wav, _ = sf.read(str(config.PROJECT_ROOT / r.segment_path))
        noisy = transforms[env](samples=wav.astype(np.float32), sample_rate=config.SAMPLE_RATE)
        noisy = normalize_volume(noisy, target_peak=config.NOISE_MIX_TARGET_PEAK)

        rel = (config.PROJECT_ROOT / r.segment_path).relative_to(config.INTERIM_DIR)
        out = config.AUG_DIR / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, noisy, config.SAMPLE_RATE, subtype="PCM_16")

        rows.append({
            "segment_path": out.relative_to(config.PROJECT_ROOT).as_posix(),
            "label": int(r.label),
            "class_name": r.class_name,
            # Same source_recording as the original, so the split treats them as one unit.
            "source_recording": r.source_recording,
            "source_folder": r.source_folder,
            "dataset": r.dataset,
        })

    df = pd.DataFrame(rows)
    df.to_csv(config.AUG_MANIFEST, index=False)

    print(f"\n{len(df)} noisy training copies -> {config.AUG_DIR}")
    print(f"SNR drawn uniformly from {lo:.0f}-{hi:.0f} dB, round-robin over {len(envs)} environments.")
    print(f"Manifest -> {config.AUG_MANIFEST}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build noise-augmented training data")
    parser.add_argument("--limit", type=int, default=None, help="only N train segments (smoke test)")
    args = parser.parse_args()
    build_augmented_train(args.limit)


if __name__ == "__main__":
    main()
