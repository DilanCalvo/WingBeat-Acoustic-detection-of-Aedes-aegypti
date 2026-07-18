# Stage 1-2: read the raw audio and cut it into fixed-length segments.

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

from . import config

MANIFEST_COLUMNS = [
    "segment_path", "label", "class_name", "source_recording", "source_folder", "dataset",
]


@dataclass
class Recording:
    audio_path: Path
    label: int
    source_recording: str  # grouping key (anti-leakage): dataset/<parent dir>
    source_folder: str     # diversity bucket: dataset/<top-level class folder>
    dataset: str


# Core audio operations
def load_audio_mono(path: Path, sample_rate: int = config.SAMPLE_RATE) -> np.ndarray:
    # Load any audio file as mono float32 at `sample_rate`.
    waveform, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return waveform.astype(np.float32)


def normalize_volume(waveform: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    # Peak-normalize (not RMS) so mixing noise later doesn't clip.
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak < 1e-9:
        return waveform
    return (waveform / peak) * target_peak


def segment_waveform(
    waveform: np.ndarray,
    segment_samples: int = config.SEGMENT_SAMPLES,
    hop_samples: int = config.SEGMENT_HOP_SAMPLES,
    min_rms: float = config.MIN_SEGMENT_RMS,
):
    # Slice into overlapping windows.
    n = waveform.size
    if n < segment_samples:
        if n == 0:
            return
        padded = np.zeros(segment_samples, dtype=np.float32)
        padded[:n] = waveform
        if _rms(padded) >= min_rms:
            yield padded
        return

    for start in range(0, n - segment_samples + 1, hop_samples):
        segment = waveform[start : start + segment_samples]
        if _rms(segment) >= min_rms:
            yield segment


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


# Raw-data discovery & labeling
def _label_for_path(relative_parts: tuple[str, ...]) -> int | None:
    # Binary label from the folder path, or None to skip the file.
    lowered = [p.lower() for p in relative_parts]
    if any(any(k in part for k in config.IGNORE_FOLDER_KEYWORDS) for part in lowered):
        return None
    if any(any(k in part for k in config.NON_AEDES_OVERRIDE_KEYWORDS) for part in lowered):
        return config.LABEL_NON_AEDES
    if any(any(k in part for k in config.AEDES_LABEL_KEYWORDS) for part in lowered):
        return config.LABEL_AEDES
    return config.LABEL_NON_AEDES


def iter_raw_recordings(dataset_dir: Path):
    # Yield a Recording for every labeled audio file in a dataset.
    dataset = dataset_dir.name
    for path in sorted(dataset_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in config.AUDIO_EXTENSIONS:
            continue
        rel = path.relative_to(dataset_dir)
        label = _label_for_path(rel.parts)
        if label is None:
            continue
        source_recording = f"{dataset}/{_group_key(rel)}"
        source_folder = f"{dataset}/{rel.parts[0]}" if len(rel.parts) > 1 else dataset
        yield Recording(path, label, source_recording, source_folder, dataset)


def _group_key(rel: Path) -> str:
    # Grouping key for one file, by how deep it's nested.
    if len(rel.parts) >= 3:          # nested: parent folder is a recording session
        return rel.parent.as_posix()
    if len(rel.parts) == 2:          # flat under a class folder: one file = one recording
        return f"{rel.parts[0]}/{rel.stem}"
    return rel.stem                  # loose file at the dataset root


# Segmentation of a single recording
def _process_recording(rec: Recording, manifest_rows: list[dict]) -> int:
    # Load, normalize and segment one recording. Returns how many were written.
    try:
        waveform = normalize_volume(load_audio_mono(rec.audio_path))
    except Exception as exc:  # noqa: BLE001 - skip unreadable files, keep going
        print(f"  [skip] {rec.audio_path.name}: {exc}")
        return 0

    class_name = config.CLASS_NAMES[rec.label]
    out_dir = config.INTERIM_DIR / class_name
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = f"{rec.source_recording}/{rec.audio_path.stem}".replace("/", "__")

    written = 0
    for idx, segment in enumerate(segment_waveform(waveform)):
        seg_path = out_dir / f"{safe_stem}__{idx:04d}.wav"
        sf.write(seg_path, segment, config.SAMPLE_RATE, subtype="PCM_16")
        manifest_rows.append(
            {
                "segment_path": seg_path.relative_to(config.PROJECT_ROOT).as_posix(),
                "label": rec.label,
                "class_name": class_name,
                "source_recording": rec.source_recording,
                "source_folder": rec.source_folder,
                "dataset": rec.dataset,
            }
        )
        written += 1
    return written


def _fill_to_target(records: list[Recording], seg_target: int | None,
                    manifest_rows: list[dict], desc: str) -> int:
    # Process records until `seg_target` segments are written.
    written = 0
    bar = tqdm(total=seg_target, desc=desc, unit="seg")
    while records and (seg_target is None or written < seg_target):
        rec = records.pop()
        n = _process_recording(rec, manifest_rows)
        written += n
        bar.update(n)
    bar.close()
    return written


# Pipeline
def run(datasets: list[str], target_per_class: int | None) -> Path:
    # Preprocess the given datasets into a balanced segment manifest.
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.SEED)

    # 1. Discover every labeled recording across the requested datasets.
    records: list[Recording] = []
    for dataset in datasets:
        dataset_dir = config.RAW_DIR / dataset
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
        records.extend(iter_raw_recordings(dataset_dir))
    if not records:
        raise RuntimeError(f"No audio found under {config.RAW_DIR}")

    positives = [r for r in records if r.label == config.LABEL_AEDES]
    negatives = [r for r in records if r.label == config.LABEL_NON_AEDES]
    print(f"Discovered {len(records)} recordings "
          f"({len(positives)} aedes / {len(negatives)} non-aedes).")

    manifest_rows: list[dict] = []

    # 2. Positives: fill to target from a shuffled pool.
    rng.shuffle(positives)
    n_pos = _fill_to_target(positives, target_per_class, manifest_rows, "aedes      ")

    # 3. Negatives: spread the target evenly across source folders for diversity.
    if target_per_class is None:
        rng.shuffle(negatives)
        n_neg = _fill_to_target(negatives, None, manifest_rows, "non_aedes  ")
    else:
        buckets: dict[str, list[Recording]] = defaultdict(list)
        for r in negatives:
            buckets[r.source_folder].append(r)
        for recs in buckets.values():
            rng.shuffle(recs)
        per_bucket = max(1, target_per_class // len(buckets))

        n_neg = 0
        # Pass 1: equal share per bucket.
        for recs in buckets.values():
            remaining = target_per_class - n_neg
            if remaining <= 0:
                break
            n_neg += _fill_to_target(recs, min(per_bucket, remaining), manifest_rows, "non_aedes p1")
        # Pass 2: top up from buckets that still have leftovers.
        for recs in buckets.values():
            remaining = target_per_class - n_neg
            if remaining <= 0:
                break
            n_neg += _fill_to_target(recs, remaining, manifest_rows, "non_aedes p2")

    # 4. Write manifest.
    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    manifest.to_csv(config.SEGMENTS_MANIFEST, index=False)

    print(f"\nDone. {len(manifest)} segments  (aedes={n_pos}  non_aedes={n_neg}).")
    print(f"Distinct source recordings (groups): {manifest['source_recording'].nunique()}")
    print("Segments per source folder:")
    print(manifest.groupby(["class_name", "source_folder"]).size().to_string())
    print(f"\nManifest: {config.SEGMENTS_MANIFEST}")
    return config.SEGMENTS_MANIFEST


def _discover_datasets() -> list[str]:
    if not config.RAW_DIR.is_dir():
        return []
    return sorted(p.name for p in config.RAW_DIR.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="WINGBEAT preprocessing")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="single dataset folder under data/raw/")
    group.add_argument("--all", action="store_true", help="process every dataset")
    parser.add_argument(
        "--target-per-class", type=int, default=config.TARGET_SEGMENTS_PER_CLASS,
        help="balanced segments per class (use 0 for everything)",
    )
    args = parser.parse_args()

    datasets = _discover_datasets() if args.all else [args.dataset]
    if not datasets:
        parser.error(f"no datasets found under {config.RAW_DIR}")
    target = None if args.target_per_class == 0 else args.target_per_class
    run(datasets, target)


if __name__ == "__main__":
    main()
