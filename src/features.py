# Stage 3: turn each segment into the two representations we need.

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config


def mel_spectrogram(waveform: np.ndarray) -> np.ndarray:
    # Log-Mel spectrogram (N_MELS, n_frames) for the CNN.
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        n_mels=config.N_MELS,
        fmin=config.FMIN,
        fmax=config.FMAX,
        power=2.0,
    )
    return librosa.power_to_db(mel, ref=np.max).astype(np.float32)


def normalize_mel(mel_db: np.ndarray, db_floor: float = 80.0) -> np.ndarray:
    # Scale a log-Mel spectrogram (dB, max=0) into [0, 1].
    return np.clip((mel_db + db_floor) / db_floor, 0.0, 1.0).astype(np.float32)


def mfcc_vector(waveform: np.ndarray) -> np.ndarray:
    # Fixed-length MFCC vector for the RandomForest: mean/std of MFCCs and derivatives.
    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=config.SAMPLE_RATE,
        n_mfcc=config.N_MFCC,
        n_fft=config.N_FFT,
        hop_length=config.HOP_LENGTH,
        fmin=config.FMIN,
        fmax=config.FMAX,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    parts = []
    for block in (mfcc, delta, delta2):
        parts.append(block.mean(axis=1))
        parts.append(block.std(axis=1))
    return np.concatenate(parts).astype(np.float32)


def build_feature_matrices(manifest_path: Path = config.SEGMENTS_MANIFEST, augmented: bool = False):
    # Read a segment manifest and write the feature arrays to data/processed/.
    manifest = pd.read_csv(manifest_path)
    if manifest.empty:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    # Map each distinct source recording to an integer group id for the split.
    group_ids = {rec: i for i, rec in enumerate(manifest["source_recording"].unique())}

    x_mfcc, x_mel, y, groups, folders = [], [], [], [], []
    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="features"):
        waveform, _ = librosa.load(
            str(config.PROJECT_ROOT / row.segment_path), sr=config.SAMPLE_RATE, mono=True
        )
        x_mfcc.append(mfcc_vector(waveform))
        x_mel.append(mel_spectrogram(waveform))
        y.append(int(row.label))
        groups.append(group_ids[row.source_recording])
        folders.append(row.source_folder)

    x_mfcc = np.stack(x_mfcc)
    x_mel = np.stack(x_mel)[..., np.newaxis]  # add channel dim for the CNN
    y = np.asarray(y, dtype=np.int64)
    groups = np.asarray(groups, dtype=np.int64)
    # Saved with the features: the split and the holdout selection both read it.
    folders = np.asarray(folders, dtype=object)

    if augmented:
        # The augmented copies are 1:1 with the train segments, so only save the arrays.
        np.save(config.AUG_MFCC_PATH, x_mfcc)
        np.save(config.AUG_MEL_PATH, x_mel)
        np.save(config.AUG_LABELS_PATH, y)
        print(f"\nAugmented MFCC: {x_mfcc.shape} -> {config.AUG_MFCC_PATH.name}")
        print(f"Augmented Mel : {x_mel.shape} -> {config.AUG_MEL_PATH.name}")
        print(f"Labels        : {y.shape}  positives={int(y.sum())}")
        return x_mfcc, x_mel, y, groups, folders

    np.save(config.MFCC_MATRIX_PATH, x_mfcc)
    np.save(config.MEL_MATRIX_PATH, x_mel)
    np.save(config.LABELS_PATH, y)
    np.save(config.GROUPS_PATH, groups)
    np.save(config.FOLDERS_PATH, folders)

    print(f"\nMFCC matrix : {x_mfcc.shape} -> {config.MFCC_MATRIX_PATH.name}")
    print(f"Mel matrix  : {x_mel.shape} -> {config.MEL_MATRIX_PATH.name}")
    print(f"Labels      : {y.shape}  positives={int(y.sum())}")
    print(f"Groups      : {len(set(groups.tolist()))} distinct recordings")
    print(f"Folders     : {len(set(folders.tolist()))} source folders -> {config.FOLDERS_PATH.name}")
    return x_mfcc, x_mel, y, groups, folders


def main() -> None:
    parser = argparse.ArgumentParser(description="WINGBEAT feature extraction")
    parser.add_argument(
        "--manifest",
        default=None,
        help="segment manifest produced by preprocess.py",
    )
    parser.add_argument(
        "--augmented", action="store_true",
        help="extract the noise-augmented training copies (config.AUG_MANIFEST) instead",
    )
    args = parser.parse_args()
    default = config.AUG_MANIFEST if args.augmented else config.SEGMENTS_MANIFEST
    build_feature_matrices(Path(args.manifest) if args.manifest else default, args.augmented)


if __name__ == "__main__":
    main()
