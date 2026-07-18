# Single inference path for the demo, and the one the tests can pin.

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from audiomentations import AddBackgroundNoise

from . import config
from .build_hybrid_testset import list_cr_noise_files, make_noisy_copy
from .features import mel_spectrogram, mfcc_vector, normalize_mel
from .preprocess import load_audio_mono, normalize_volume, segment_waveform

# Which feature representation each model consumes.
MODEL_KIND_CNN = "cnn"
MODEL_KIND_RF = "baseline"

# The serialized models the demo can show, and the kind each one consumes.
MODEL_PATHS: dict[str, Path] = {
    "cnn": config.CNN_MODEL_PATH,
    "cnn_aug": config.CNN_AUG_MODEL_PATH,
    "baseline": config.BASELINE_MODEL_PATH,
}
MODEL_KIND: dict[str, str] = {
    "cnn": MODEL_KIND_CNN,
    "cnn_aug": MODEL_KIND_CNN,
    "baseline": MODEL_KIND_RF,
}


@dataclass
class PredictionResult:
    # One model's verdict on one clip.

    segment_probs: np.ndarray  # (n_seg,) P(aedes) per 1 s window
    aggregate: float           # clip-level confidence = mean(segment_probs); nan if empty
    verdict: int               # 1 if aggregate >= threshold else 0
    n_segments: int


# Models
def load_all_models() -> dict[str, object]:
    # Load every serialized model present under models/.
    models: dict[str, object] = {}
    for name, path in MODEL_PATHS.items():
        if not path.exists():
            continue
        if name == "baseline":
            import joblib

            models[name] = joblib.load(path)
        else:
            from tensorflow import keras

            models[name] = keras.models.load_model(path)
    return models


# Audio in
def prepare_waveform(source) -> np.ndarray:
    # Normalized 8 kHz mono waveform from a path or an uploaded file.
    if isinstance(source, (str, Path)):
        waveform = load_audio_mono(Path(source))
    else:
        raw = source.read() if hasattr(source, "read") else source
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if data.ndim == 2:  # stereo -> mono
            data = data.mean(axis=1)
        if sr != config.SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        waveform = data.astype(np.float32)
    return normalize_volume(waveform, target_peak=0.95)


def add_cr_noise(waveform: np.ndarray, environment: str, snr_db: float,
                 seed: int = config.SEED) -> np.ndarray:
    # Mix real CR noise onto a clip at a fixed SNR.
    test_files = [str(p) for p in list_cr_noise_files()[environment]["test"]]
    transform = AddBackgroundNoise(
        sounds_path=test_files,
        min_snr_db=snr_db,
        max_snr_db=snr_db,
        noise_rms="relative",
        p=1.0,
    )
    random.seed(seed)  # audiomentations draws from the global RNG
    return make_noisy_copy(waveform, transform)


# Inference
def count_segments(waveform: np.ndarray) -> int:
    # How many 1 s windows this clip yields (0 = too short or silent).
    return sum(1 for _ in segment_waveform(waveform))


def predict_waveform(waveform: np.ndarray, model, model_kind: str,
                     threshold: float = config.DECISION_THRESHOLD) -> PredictionResult:
    # Score a clip window by window and average into one verdict.
    segments = list(segment_waveform(waveform))
    if not segments:
        return PredictionResult(np.empty(0, dtype=np.float32), float("nan"), 0, 0)

    if model_kind == MODEL_KIND_CNN:
        mel = np.stack([normalize_mel(mel_spectrogram(s)) for s in segments])[..., np.newaxis]
        probs = model.predict(mel, verbose=0).ravel()
    else:
        feats = np.stack([mfcc_vector(s) for s in segments])
        probs = model.predict_proba(feats)[:, 1]

    probs = probs.astype(np.float32)
    aggregate = float(probs.mean())
    return PredictionResult(probs, aggregate, int(aggregate >= threshold), len(segments))


def predict_all(waveform: np.ndarray, models: dict[str, object],
                threshold: float = config.DECISION_THRESHOLD) -> dict[str, PredictionResult]:
    # Run every loaded model over one clip.
    return {
        name: predict_waveform(waveform, model, MODEL_KIND[name], threshold)
        for name, model in models.items()
    }
