# Checks for the demo inference path (src/inference.py).

from __future__ import annotations

import numpy as np
import pytest

from src import config
from src.features import mel_spectrogram, normalize_mel
from src.preprocess import segment_waveform

SAMPLES_DIR = config.PROJECT_ROOT / "demo" / "samples"

needs_cnn = pytest.mark.skipif(
    not config.CNN_MODEL_PATH.exists(), reason="train the CNN first (models/cnn_wingbeat.h5)"
)
needs_samples = pytest.mark.skipif(
    not any(SAMPLES_DIR.glob("*.wav")),
    reason="run `python -m scripts.make_demo_samples` first",
)


def _first(glob: str):
    files = sorted(SAMPLES_DIR.glob(glob))
    if not files:
        pytest.skip(f"no demo sample matching {glob}")
    return files[0]


@needs_cnn
@needs_samples
def test_demo_cnn_path_applies_normalize_mel():
    # The demo's prob must match the normalized-Mel reference, not the raw-dB one.
    tf = pytest.importorskip("tensorflow")  # noqa: F841
    from src import inference

    cnn = inference.load_all_models()["cnn"]
    wav = inference.prepare_waveform(_first("neg_*.wav"))
    seg0 = list(segment_waveform(wav))[0]

    mel_raw = mel_spectrogram(seg0)
    ref_norm = float(cnn.predict(normalize_mel(mel_raw)[np.newaxis, ..., np.newaxis],
                                 verbose=0).ravel()[0])
    ref_rawdb = float(cnn.predict(mel_raw[np.newaxis, ..., np.newaxis], verbose=0).ravel()[0])

    demo_prob = float(inference.predict_waveform(wav, cnn, inference.MODEL_KIND_CNN).segment_probs[0])

    # Sanity: this sample actually discriminates (normalize_mel changes the answer here).
    assert abs(ref_norm - ref_rawdb) > 0.05, "chosen sample does not exercise the footgun"

    assert demo_prob == pytest.approx(ref_norm, abs=1e-4), (
        "demo CNN prob drifted from the normalized-Mel reference; the training-time "
        "preprocessing and the demo's have diverged"
    )
    assert abs(demo_prob - ref_rawdb) > 0.05, (
        "demo prob matches the raw-dB (un-normalized) output, so normalize_mel is "
        "missing from the demo path"
    )


@needs_cnn
@needs_samples
def test_demo_narrative_holds_on_clean_audio():
    # A curated positive is detected on clean audio; a curated negative is not.
    pytest.importorskip("tensorflow")
    from src import inference

    cnn = inference.load_all_models()["cnn"]
    pos = inference.predict_waveform(inference.prepare_waveform(_first("pos_*.wav")),
                                     cnn, inference.MODEL_KIND_CNN)
    neg = inference.predict_waveform(inference.prepare_waveform(_first("neg_*.wav")),
                                     cnn, inference.MODEL_KIND_CNN)
    assert pos.verdict == 1 and pos.aggregate >= config.DECISION_THRESHOLD
    assert neg.verdict == 0 and neg.aggregate < config.DECISION_THRESHOLD


@needs_samples
def test_live_noise_never_uses_the_reserved_recording():
    # The live noise toggle mixes only the test pool.
    from src.build_hybrid_testset import list_cr_noise_files

    pools = list_cr_noise_files()
    for env in config.CR_NOISE_ENVIRONMENTS:
        test_names = {p.name for p in pools[env]["test"]}
        assert test_names, f"no test-pool noise for {env}"
        assert not any(n.endswith("_03.wav") for n in test_names), (
            f"{env}: reserved *_03 leaked into the demo's live-noise pool"
        )


@needs_samples
def test_short_or_silent_clip_yields_no_segments():
    # A silent or too-short clip must give n_segments=0 so the UI can warn.
    from src import inference

    silent = np.zeros(config.SAMPLE_RATE // 4, dtype=np.float32)
    assert inference.count_segments(silent) == 0
