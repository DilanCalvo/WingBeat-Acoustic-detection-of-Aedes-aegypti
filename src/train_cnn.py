# Stage 4b: small CNN on log-Mel spectrograms (the main model).

from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

from . import config
from .data_split import grouped_split
from .features import normalize_mel
from .train_baseline import false_negative_rate, per_folder_report


def load_split():
    # Load the Mel features with the same grouped split as the baseline.
    x = normalize_mel(np.load(config.MEL_MATRIX_PATH))
    y = np.load(config.LABELS_PATH)
    groups = np.load(config.GROUPS_PATH)
    folders = np.load(config.FOLDERS_PATH, allow_pickle=True)
    train_idx, val_idx, test_idx = grouped_split(groups, folders)
    return x, y, folders, train_idx, val_idx, test_idx


def load_augmented(train_idx):
    # The noisy twins of the train segments.
    if not config.AUG_MEL_PATH.exists():
        raise FileNotFoundError(
            f"No augmented features at {config.AUG_MEL_PATH}. Run:\n"
            f"    python -m src.build_noise_augmentation\n"
            f"    python -m src.features --augmented"
        )
    x_aug = normalize_mel(np.load(config.AUG_MEL_PATH))
    y_aug = np.load(config.AUG_LABELS_PATH)
    # If these drift apart the model trains on mislabeled audio and still looks fine.
    assert len(x_aug) == len(train_idx), (
        f"augmented set has {len(x_aug)} rows but train has {len(train_idx)} — "
        f"re-run build_noise_augmentation + features --augmented"
    )
    return x_aug, y_aug


def build_cnn(input_shape, n_classes: int = 2):
    # 3x (Conv-BN-Pool) -> global pooling -> dense. Sigmoid for binary.
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="wingbeat_cnn",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def evaluate(model, x_test, y_test) -> dict:
    # Clean-condition metrics at the 0.5 threshold.
    proba = model.predict(x_test, verbose=0).ravel()
    y_pred = (proba >= 0.5).astype(int)
    metrics = {
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "false_negative_rate": false_negative_rate(y_test, y_pred),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "n_test": int(len(y_test)),
        "n_test_positives": int(y_test.sum()),
    }
    print("\n--- CNN (clean test) ---")
    print(classification_report(y_test, y_pred, target_names=config.CLASS_NAMES, zero_division=0))
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}   FN rate: {metrics['false_negative_rate']:.3f}")
    return metrics


def _print_comparison(cnn_metrics: dict) -> None:
    path = config.RESULTS_DIR / "baseline_clean_metrics.json"
    if not path.exists():
        return
    base = json.loads(path.read_text())
    print("\n=== Baseline vs CNN (clean) ===")
    print(f"{'metric':<22}{'RF baseline':>14}{'CNN':>10}")
    for key in ("f1", "precision", "recall", "false_negative_rate"):
        print(f"{key:<22}{base.get(key, float('nan')):>14.3f}{cnn_metrics[key]:>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Mel-spectrogram CNN")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augment", action="store_true",
                        help="ALSO train on noise-augmented copies of the training segments "
                             "(the mitigation arm). Writes to models/cnn_wingbeat_aug.h5.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import tensorflow as tf

    tf.keras.utils.set_random_seed(config.SEED)

    out_path = args.out or str(config.CNN_AUG_MODEL_PATH if args.augment else config.CNN_MODEL_PATH)
    x, y, folders, train_idx, val_idx, test_idx = load_split()
    input_shape = x.shape[1:]
    print(f"Input shape: {input_shape}   train/val/test = "
          f"{len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    x_train, y_train = x[train_idx], y[train_idx]
    if args.augment:
        # Train on clean + noisy.
        x_aug, y_aug = load_augmented(train_idx)
        x_train = np.concatenate([x_train, x_aug])
        y_train = np.concatenate([y_train, y_aug])
        del x_aug
        print(f"AUGMENTED: {len(train_idx)} clean + {len(y_aug)} noisy = {len(y_train)} "
              f"training samples (SNR {config.AUG_SNR_RANGE_DB[0]:.0f}-"
              f"{config.AUG_SNR_RANGE_DB[1]:.0f} dB, reserved noise only)")

    model = build_cnn(input_shape)
    model.summary()

    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: float(weights[0]), 1: float(weights[1])}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    model.fit(
        x_train, y_train,
        validation_data=(x[val_idx], y[val_idx]),
        epochs=args.epochs, batch_size=args.batch_size,
        class_weight=class_weight, callbacks=callbacks, verbose=2,
    )

    metrics = evaluate(model, x[test_idx], y[test_idx])
    y_pred = (model.predict(x[test_idx], verbose=0).ravel() >= 0.5).astype(int)
    per_folder_report(y[test_idx], y_pred, folders[test_idx])

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    name = "cnn_aug_clean_metrics.json" if args.augment else "cnn_clean_metrics.json"
    (config.RESULTS_DIR / name).write_text(json.dumps(metrics, indent=2))
    if not args.augment:
        _print_comparison(metrics)
    print(f"\nModel  -> {out_path}")


if __name__ == "__main__":
    main()
