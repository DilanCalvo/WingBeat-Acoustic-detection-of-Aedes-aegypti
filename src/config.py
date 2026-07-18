# Central configuration.

from __future__ import annotations

from pathlib import Path

# Reproducibility
SEED: int = 42  # numpy, scikit-learn, TensorFlow and all splits

# Paths (relative to the repo root, so it works on local + Colab)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"            # untouched public datasets
NOISE_CR_DIR: Path = DATA_DIR / "ruido_costarica"  # team-recorded CR noise
INTERIM_DIR: Path = DATA_DIR / "interim"    # segmented, resampled WAV chunks
PROCESSED_DIR: Path = DATA_DIR / "processed"  # feature arrays + split manifests
HYBRID_DIR: Path = DATA_DIR / "hybrid"      # clean mosquito + CR noise @ SNR

# Manifest + feature artifacts (written by preprocess.py / features.py).
SEGMENTS_MANIFEST: Path = PROCESSED_DIR / "segments.csv"
MFCC_MATRIX_PATH: Path = PROCESSED_DIR / "X_mfcc.npy"
MEL_MATRIX_PATH: Path = PROCESSED_DIR / "X_mel.npy"
LABELS_PATH: Path = PROCESSED_DIR / "y.npy"
GROUPS_PATH: Path = PROCESSED_DIR / "groups.npy"
# "<dataset>/<class folder>" per segment.
FOLDERS_PATH: Path = PROCESSED_DIR / "folders.npy"

MODELS_DIR: Path = PROJECT_ROOT / "models"
RESULTS_DIR: Path = PROJECT_ROOT / "results"

# Serialized models (committed so the demo runs without retraining).
BASELINE_MODEL_PATH: Path = MODELS_DIR / "rf_baseline.pkl"
CNN_MODEL_PATH: Path = MODELS_DIR / "cnn_wingbeat.h5"

# 8 kHz is the native rate of all three datasets, so nothing is lost resampling.
SAMPLE_RATE: int = 8_000

SEGMENT_SECONDS: float = 1.0     # fixed-length analysis window
SEGMENT_OVERLAP: float = 0.5     # 50% sliding-window overlap
SEGMENT_SAMPLES: int = int(SEGMENT_SECONDS * SAMPLE_RATE)
SEGMENT_HOP_SAMPLES: int = int(SEGMENT_SAMPLES * (1.0 - SEGMENT_OVERLAP))

# Drop near-silent segments (RMS below this) so we don't train on empty air.
MIN_SEGMENT_RMS: float = 1e-3

# Feature extraction (shared STFT settings -> Mel for CNN, MFCC for RF)
N_FFT: int = 512        # ~64 ms window at 8 kHz
HOP_LENGTH: int = 128   # ~16 ms hop -> ~59 frames per 1 s segment
FMIN: float = 0.0
FMAX: float = SAMPLE_RATE / 2  # Nyquist

# Mel-spectrogram (CNN input image)
N_MELS: int = 64

# MFCC (RandomForest input).
N_MFCC: int = 20

# Labels — binary task: Aedes aegypti vs. everything else
CLASS_NAMES: tuple[str, str] = ("non_aedes", "aedes_aegypti")
LABEL_NON_AEDES: int = 0
LABEL_AEDES: int = 1

# Layout expected by preprocess.py: data/raw/<dataset>/<class_folder>/<file>.wav
AEDES_LABEL_KEYWORDS: tuple[str, ...] = (
    "aedes_aegypti",
    "aedes aegypti",
    "ae_aegypti",
    "ae. aegypti",
    "aegypti",
)
# Checked first: a match here forces label 0 (e.g. UFRGS "No_Aedes_aegypti").
NON_AEDES_OVERRIDE_KEYWORDS: tuple[str, ...] = (
    "no_aedes",
    "non_aedes",
    "no aedes",
    "not aedes",
)
# Folder names to ignore entirely (metadata, licenses, etc.).
IGNORE_FOLDER_KEYWORDS: tuple[str, ...] = ("metadata", "license", "readme", "doc")
# Audio extensions preprocess.py will ingest.
AUDIO_EXTENSIONS: tuple[str, ...] = (".wav", ".flac", ".ogg", ".mp3")

# Train / val / test split Grouped by source recording, never by segment: two segments.
TEST_FRACTION: float = 0.20
VAL_FRACTION: float = 0.15   # fraction of the remaining train pool

# Also stratified by source_folder (species x dataset): the grouped split runs inside.
STRATIFY_SPLIT_BY_FOLDER: bool = True

# Datasets kept out of train/val/test entirely, as an independent test set.
HOLDOUT_DATASETS: tuple[str, ...] = ("ufrgs",)

# Balanced subsampling (preprocess) Segments per class for the balanced subset.
TARGET_SEGMENTS_PER_CLASS: int | None = 15_000

# RandomForest baseline
RF_N_ESTIMATORS: int = 300
RF_CLASS_WEIGHT: str = "balanced"  # positives are the minority

# Central experiment — noise robustness at controlled SNR.
SNR_LEVELS_DB: dict[str, float] = {
    "easy": 20.0,    # noise clearly in the background
    "medium": 10.0,  # noise comparable to signal
    "hard": 0.0,     # noise as loud as the wingbeat
}

# The four evaluation conditions reported in results/degradation_table.md.
EVAL_CONDITIONS: tuple[str, ...] = ("clean", "easy", "medium", "hard")

# The five CR environments recorded for the hybrid test set.
CR_NOISE_ENVIRONMENTS: tuple[str, ...] = (
    "street_traffic",
    "kitchen_fan",
    "tv_room",
    "park_green",
    "voices_market",
)

# Per environment, reserve this many recordings (the highest-numbered ones) for possible.
NOISE_FILES_HELD_FOR_TRAIN: int = 1

# Mixing can push the peak above 1.0, so we normalize back to this to avoid clipping.
NOISE_MIX_TARGET_PEAK: float = 0.95

# Manifest written by build_hybrid_testset.py, read by evaluate.py.
HYBRID_MANIFEST: Path = HYBRID_DIR / "hybrid_manifest.csv"

# Evaluation the degradation measurement) Deployed operating point.
DECISION_THRESHOLD: float = 0.5

# F1 confidence interval by bootstrap.
BOOTSTRAP_RESAMPLES: int = 1_000

# Aedes wingbeat fundamental.
WINGBEAT_BAND_HZ: tuple[float, float] = (400.0, 700.0)

# Cached per-segment probabilities (gitignored, regenerable).
HYBRID_PREDICTIONS: Path = RESULTS_DIR / "hybrid_predictions.csv"

# Main results table.
DEGRADATION_TABLE: Path = RESULTS_DIR / "degradation_table.md"
DEGRADATION_METRICS: Path = RESULTS_DIR / "degradation_metrics.json"

# Mitigation arm recalibration + noise-augmented retraining The reserved noise recording.
NOISE_POOL_DIR: Path = INTERIM_DIR / "noise_pool"   # regenerable -> gitignored
NOISE_AUG_SPLIT: float = 0.5

# Noisy validation set: val segments + calibration-half noise. Never touches test.
HYBRID_VAL_DIR: Path = DATA_DIR / "hybrid_val"
HYBRID_VAL_MANIFEST: Path = HYBRID_VAL_DIR / "hybrid_val_manifest.csv"

# Noise-augmented copies of the TRAIN segments (one per segment).
AUG_DIR: Path = DATA_DIR / "augmented"
AUG_MANIFEST: Path = AUG_DIR / "aug_manifest.csv"
AUG_MFCC_PATH: Path = PROCESSED_DIR / "X_mfcc_aug.npy"
AUG_MEL_PATH: Path = PROCESSED_DIR / "X_mel_aug.npy"
AUG_LABELS_PATH: Path = PROCESSED_DIR / "y_aug.npy"

# Augmentation SNR is drawn uniformly from this range rather than the three fixed test.
AUG_SNR_RANGE_DB: tuple[float, float] = (0.0, 20.0)

CNN_AUG_MODEL_PATH: Path = MODELS_DIR / "cnn_wingbeat_aug.h5"

# Thresholds chosen on the noisy validation set, written by calibrate.py.
CALIBRATION_PATH: Path = RESULTS_DIR / "calibration.json"
VAL_PREDICTIONS: Path = RESULTS_DIR / "val_predictions.csv"   # cache, gitignored
# Second operating point: best precision achievable at or above this recall on validation.
TARGET_RECALL: float = 0.90
