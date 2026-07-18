from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
from audiomentations import AddBackgroundNoise

from src import config
from src.build_hybrid_testset import assign_environments, list_cr_noise_files, make_noisy_copy
from src.data_split import dataset_of, grouped_split, holdout_mask
from src.features import normalize_mel
from src.preprocess import normalize_volume
from src.train_baseline import false_negative_rate

# Data-dependent tests are skipped, not failed, on a fresh clone.
needs_features = pytest.mark.skipif(
    not config.FOLDERS_PATH.exists(), reason="run src.preprocess + src.features first"
)
needs_hybrid = pytest.mark.skipif(
    not config.HYBRID_MANIFEST.exists(), reason="run src.build_hybrid_testset first"
)
needs_noise = pytest.mark.skipif(
    not any(config.NOISE_CR_DIR.glob("*.wav")), reason="no CR noise recordings present"
)


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    clean = clean.astype(np.float64)
    noisy = noisy.astype(np.float64)
    k = float(clean @ noisy) / float(clean @ clean)
    residual = noisy - k * clean
    return float(20 * np.log10(np.linalg.norm(k * clean) / (np.linalg.norm(residual) + 1e-12)))


# The central experiment: is the noise actually mixed at the SNR we claim?
@pytest.mark.parametrize("snr_db", [20.0, 10.0, 0.0])
def test_make_noisy_copy_realizes_the_requested_snr(tmp_path, snr_db):
    rng = np.random.default_rng(0)
    noise_file = tmp_path / "noise.wav"
    sf.write(noise_file, rng.normal(0, 0.1, config.SAMPLE_RATE * 4).astype(np.float32),
             config.SAMPLE_RATE, subtype="PCM_16")

    t = np.arange(config.SEGMENT_SAMPLES) / config.SAMPLE_RATE
    clean = (0.5 * np.sin(2 * np.pi * 550 * t)).astype(np.float32)   # a fake "wingbeat"

    transform = AddBackgroundNoise(
        sounds_path=[str(noise_file)], min_snr_db=snr_db, max_snr_db=snr_db,
        noise_rms="relative", p=1.0,
    )
    noisy = make_noisy_copy(clean, transform)

    assert not np.allclose(noisy, clean), "the noise was not applied at all"
    assert measured_snr_db(clean, noisy) == pytest.approx(snr_db, abs=0.5)


def test_noise_is_always_applied_never_probabilistic(tmp_path):
    rng = np.random.default_rng(1)
    noise_file = tmp_path / "n.wav"
    sf.write(noise_file, rng.normal(0, 0.1, config.SAMPLE_RATE * 2).astype(np.float32),
             config.SAMPLE_RATE, subtype="PCM_16")
    t = np.arange(config.SEGMENT_SAMPLES) / config.SAMPLE_RATE
    clean = (0.5 * np.sin(2 * np.pi * 550 * t)).astype(np.float32)
    transform = AddBackgroundNoise(sounds_path=[str(noise_file)], min_snr_db=10, max_snr_db=10,
                                   noise_rms="relative", p=1.0)
    for _ in range(20):
        assert not np.allclose(make_noisy_copy(clean, transform), clean)


def test_normalize_volume_preserves_snr():
    rng = np.random.default_rng(2)
    clean = rng.normal(0, 0.3, config.SEGMENT_SAMPLES)
    noise = rng.normal(0, 0.1, config.SEGMENT_SAMPLES)
    before = measured_snr_db(clean, clean + noise)
    after = measured_snr_db(clean, normalize_volume(clean + noise, target_peak=0.95))
    assert after == pytest.approx(before, abs=1e-6)


def test_normalize_volume_never_clips():
    loud = np.array([-3.0, 2.5, 0.1], dtype=np.float32)
    assert np.max(np.abs(normalize_volume(loud, target_peak=0.95))) == pytest.approx(0.95)


# Noise pools: the test noise must never reach the model or the threshold
@needs_noise
def test_noise_pools_are_disjoint():
    # Test noise (_01,_02) vs reserved noise (_03); they must not overlap.
    pools = list_cr_noise_files()
    for env, pool in pools.items():
        test_files = {p.name for p in pool["test"]}
        train_files = {p.name for p in pool["train"]}
        assert test_files, f"{env}: no test noise"
        assert train_files, f"{env}: nothing reserved for training"
        assert not (test_files & train_files), f"{env}: the same recording is in both pools"


@needs_noise
def test_aug_and_cal_halves_are_disjoint_and_not_silent():
    # Split in time, so augmentation must not reuse the calibration audio.
    from src.build_noise_augmentation import prepare_noise_pools

    pools = prepare_noise_pools()
    for env in config.CR_NOISE_ENVIRONMENTS:
        aug, cal = pools["aug"][env], pools["cal"][env]
        assert aug and cal, f"{env}: a half is missing"
        for half in (aug, cal):
            for path in half:
                wav, _ = sf.read(str(path))
                assert np.sqrt(np.mean(wav ** 2)) > config.MIN_SEGMENT_RMS, f"{path} is silent"
        # Different halves of the same source file -> different audio content.
        a, _ = sf.read(str(aug[0]))
        c, _ = sf.read(str(cal[0]))
        n = min(len(a), len(c))
        assert not np.allclose(a[:n], c[:n]), f"{env}: aug and cal halves are the same audio"


# The split: the anti-leakage guarantee
@needs_features
def test_split_never_leaks_a_recording():
    groups = np.load(config.GROUPS_PATH)
    folders = np.load(config.FOLDERS_PATH, allow_pickle=True)
    tr, va, te = grouped_split(groups, folders)
    g_tr, g_va, g_te = (set(groups[i].tolist()) for i in (tr, va, te))
    assert g_tr.isdisjoint(g_te)
    assert g_tr.isdisjoint(g_va)
    assert g_va.isdisjoint(g_te)


@needs_features
def test_split_excludes_the_holdout_dataset():
    # UFRGS is the independent device-shift test, it can't be in train/val/test.
    groups = np.load(config.GROUPS_PATH)
    folders = np.load(config.FOLDERS_PATH, allow_pickle=True)
    tr, va, te = grouped_split(groups, folders)
    for idx in (tr, va, te):
        assert not holdout_mask(folders[idx]).any()
    assert holdout_mask(folders).sum() > 0, "the holdout vanished entirely"


@needs_features
def test_every_source_folder_appears_in_train_and_test():
    # The bug that started the audit: humbugdb/background landed 100% in train.
    groups = np.load(config.GROUPS_PATH)
    folders = np.load(config.FOLDERS_PATH, allow_pickle=True)
    tr, _, te = grouped_split(groups, folders)
    expected = set(folders[~holdout_mask(folders)].tolist())
    assert set(folders[tr].tolist()) == expected
    assert set(folders[te].tolist()) == expected
    assert "humbugdb/background" in set(folders[te].tolist())


# Features: the silent train/serve skew landmine
def test_normalize_mel_maps_db_range_to_unit_interval():
    # X_mel is raw dB and the CNN trains on normalize_mel .
    mel_db = np.array([[-80.0, -40.0, 0.0]], dtype=np.float32)
    out = normalize_mel(mel_db)
    assert out.min() >= 0.0 and out.max() <= 1.0
    np.testing.assert_allclose(out, [[0.0, 0.5, 1.0]], atol=1e-6)
    # Values below the floor must clamp, not go negative.
    assert normalize_mel(np.array([[-200.0]], dtype=np.float32)).min() == 0.0


# Metrics: the guards against a fake result
def test_trivial_f1_matches_the_base_rate_formula():
    # The floor every F1 is read against: answering "aedes" to everything.
    from src.evaluate import trivial_f1

    y = np.array([1] * 483 + [0] * 517)
    p = y.mean()
    assert trivial_f1(y) == pytest.approx(2 * p / (1 + p))
    # And it really is what a predict-everything-positive classifier scores.
    from sklearn.metrics import f1_score

    assert f1_score(y, np.ones_like(y)) == pytest.approx(trivial_f1(y))

def test_false_negative_rate_counts_missed_mosquitoes():
    y = np.array([1, 1, 1, 1, 0, 0])
    pred = np.array([1, 0, 0, 1, 0, 1])     # 2 of 4 positives missed
    assert false_negative_rate(y, pred) == pytest.approx(0.5)


def test_assign_environments_is_balanced_and_deterministic():
    # No environment may end up disproportionately positive.
    labels = [1, 0] * 500
    envs = assign_environments(labels)
    assert envs == assign_environments(labels), "not deterministic"
    df = pd.DataFrame({"env": envs, "y": labels})
    counts = df.groupby("env").y.agg(["size", "sum"])
    assert counts["size"].max() - counts["size"].min() <= 1
    assert counts["sum"].max() - counts["sum"].min() <= 1


# The hybrid test set + the calibration set
@needs_hybrid
def test_hybrid_manifest_integrity():
    m = pd.read_csv(config.HYBRID_MANIFEST)
    sizes = m.groupby("condition").size()
    assert sizes.nunique() == 1, "the conditions do not hold the same number of segments"
    assert set(sizes.index) == set(config.EVAL_CONDITIONS)
    # The holdout must not have leaked into the noise experiment.
    assert not set(dataset_of(m["source_folder"].values)) & set(config.HOLDOUT_DATASETS)
    # One environment per segment, fixed across SNRs.
    assert (m.groupby("source_segment_path").environment.nunique() == 1).all()


@needs_hybrid
@pytest.mark.skipif(not config.HYBRID_VAL_MANIFEST.exists(),
                    reason="run src.build_hybrid_testset --split val first")
def test_calibration_set_never_touches_the_test_set():
    # The threshold is chosen on validation.
    val = pd.read_csv(config.HYBRID_VAL_MANIFEST)
    test = pd.read_csv(config.HYBRID_MANIFEST)
    assert not set(val["source_segment_path"]) & set(test["source_segment_path"])
    assert not set(val["source_recording"]) & set(test["source_recording"])
    test_noise = {p.name for pool in list_cr_noise_files().values() for p in pool["test"]}
    cal_noise = {p.name for p in (config.NOISE_POOL_DIR / "cal").glob("*.wav")}
    assert cal_noise, "the calibration noise half was never built"
    assert not (test_noise & cal_noise)
