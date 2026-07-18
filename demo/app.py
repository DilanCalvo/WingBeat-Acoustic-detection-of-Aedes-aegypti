# WINGBEAT offline Streamlit demo.

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no window, just render the figures

import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# So `src` is importable when Streamlit runs this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src import config, inference  # noqa: E402
from src.features import mel_spectrogram  # noqa: E402

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
FONTS_DIR = Path(__file__).resolve().parent / "static" / "fonts"

# Palette, same tokens as the theme in .streamlit/config.toml.
INK, INK_2, INK_3 = "#1A1D21", "#5B6167", "#8A9096"
LINE, ACCENT = "#E4E2DC", "#0F766E"
BAND = "#5EEAD4"  # light teal, readable over the dark magma colormap


def _use_plex_in_matplotlib() -> None:
    # Use the same font in the spectrograms as the rest of the UI.
    ttf = FONTS_DIR / "IBMPlexSans-Regular.ttf"
    if not ttf.exists():
        return
    try:
        from matplotlib import font_manager

        font_manager.fontManager.addfont(str(ttf))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(ttf)).get_name()
    except Exception:  # noqa: BLE001 - cosmetic only, shouldn't break the demo
        pass

# Model columns: (key, title, role). Labels are Spanish because they're shown in the UI.
MODEL_COLUMNS = [
    ("cnn", "CNN ingenua", "entrenada sin ruido"),
    ("cnn_aug", "CNN + ruido CR", "reentrenada · la cura"),
    ("baseline", "RandomForest", "modelo base · MFCC"),
]

# Noise slider: label -> condition key in config.SNR_LEVELS_DB (None = clean).
NOISE_LEVELS = {
    "Limpio": None,
    "20 dB": "easy",
    "10 dB": "medium",
    "0 dB": "hard",
}

# Readable species names for the sample picker.
PRETTY_SPECIES = {
    "aegypti": "Aedes aegypti",
    "albopictus": "Ae. albopictus (especie hermana)",
    "pipiens": "Culex pipiens",
    "quinquefasciatus": "Culex quinquefasciatus",
    "arabiensis": "Anopheles arabiensis",
    "gambiae": "Anopheles gambiae",
    "background": "Ruido de fondo (sin insecto)",
}

# Readable environment names.
ENV_LABELS = {
    "street_traffic": "Tráfico callejero",
    "kitchen_fan": "Ventilador de cocina",
    "tv_room": "Sala con TV",
    "park_green": "Parque / zona verde",
    "voices_market": "Mercado con voces",
}

CSS = """
<style>
:root{
  --bg:#FAFAF8; --surface:#FFFFFF; --surface-2:#F1F0EC;
  --ink:#1A1D21; --ink-2:#5B6167; --ink-3:#8A9096;
  --line:#E4E2DC; --accent:#0F766E;
  --detect:#0E7C6B; --fail:#B23A28; --neutral:#8A9096;
}
/* --- hide the dev chrome ---
   Don't hide all of [data-testid="stToolbar"]: it holds the button that reopens the
   sidebar (stExpandSidebarButton) once it's collapsed. toolbarMode "minimal" in
   config.toml already removes the Deploy/hamburger menu; here we only cover the
   rest (old menu and the "Made with Streamlit" footer). */
#MainMenu, footer, [data-testid="stDecoration"]{visibility:hidden; height:0;}
[data-testid="stHeader"]{background:transparent;}
/* --- layout --- */
.block-container{max-width:1180px; padding-top:2rem; padding-bottom:4rem;}
/* --- header --- */
.wb-head{margin-bottom:1.5rem;}
.wb-brand{font-weight:600; font-size:1.55rem; letter-spacing:.01em; color:var(--ink); line-height:1.1;}
.wb-brand .dot{color:var(--accent);}
.wb-sub{color:var(--ink-2); font-size:.95rem; margin-top:.25rem;}
.wb-rule{height:1px; background:var(--line); margin-top:1rem;}
/* --- section header --- */
.wb-section{display:flex; align-items:baseline; gap:.65rem; border-bottom:1px solid var(--line);
  padding-bottom:.5rem; margin:2rem 0 1.1rem;}
.wb-section .wb-idx{font-family:'IBM Plex Mono',monospace; color:var(--accent); font-size:.8rem; font-weight:500;}
.wb-section .wb-title{font-weight:600; font-size:1.08rem; color:var(--ink);}
.wb-section .wb-desc{color:var(--ink-3); font-size:.82rem; margin-left:auto;}
/* --- small-caps label --- */
.wb-label{text-transform:uppercase; letter-spacing:.09em; font-size:.7rem; color:var(--ink-3);
  font-weight:600; margin:.2rem 0 .35rem;}
/* --- verdict cards --- */
.wb-grid{display:grid; grid-template-columns:repeat(3,1fr); gap:1rem;}
.wb-card{background:var(--surface); border:1px solid var(--line); border-radius:9px;
  padding:1.15rem 1.2rem; display:flex; flex-direction:column; gap:.9rem; min-height:190px;}
.wb-card-head{display:flex; flex-direction:column; gap:.12rem;}
.wb-model{font-weight:600; font-size:1.02rem; color:var(--ink);}
.wb-role{font-size:.77rem; color:var(--ink-3);}
.wb-chip{display:flex; align-items:center; justify-content:center; font-family:'IBM Plex Mono',monospace;
  font-weight:500; letter-spacing:.07em; font-size:.86rem; padding:.55rem .2rem; border-radius:6px;
  border:1px solid transparent;}
.wb-chip--detect{color:#0B5D50; background:#E7F1EF; border-color:#BFE0DA;}
.wb-chip--fail{color:#8F2A1C; background:#F6E8E4; border-color:#E7C6BE;}
.wb-chip--neutral{color:var(--ink-3); background:var(--surface-2); border-color:var(--line);}
.wb-conf{display:flex; flex-direction:column; gap:.45rem;}
.wb-conf-row{display:flex; justify-content:space-between; align-items:baseline; font-size:.8rem; color:var(--ink-2);}
.wb-num{font-family:'IBM Plex Mono',monospace; color:var(--ink); font-weight:500; font-size:.92rem;}
.wb-bar{height:6px; background:var(--surface-2); border-radius:99px; overflow:hidden;}
.wb-bar-fill{height:100%; border-radius:99px;}
.wb-bar-fill--detect{background:var(--detect);}
.wb-bar-fill--fail{background:var(--fail);}
.wb-bar-fill--neutral{background:var(--neutral);}
.wb-spark{width:100%; height:32px; display:block; margin-top:.1rem;}
.wb-spark-line{fill:none; stroke:var(--accent); stroke-width:1.5; vector-effect:non-scaling-stroke;}
.wb-spark-mid{stroke:var(--line); stroke-width:1; stroke-dasharray:3 3;}
.wb-meta{font-family:'IBM Plex Mono',monospace; font-size:.71rem; color:var(--ink-3);}
/* --- empty state --- */
.wb-empty{border:1px dashed var(--line); border-radius:9px; padding:3rem 1.5rem; text-align:center;
  color:var(--ink-3); background:var(--surface); font-size:.92rem;}
/* --- footnote --- */
.wb-foot{border-top:1px solid var(--line); margin-top:2.5rem; padding-top:1rem; color:var(--ink-3);
  font-size:.8rem; line-height:1.6;}
.wb-foot code{color:var(--ink-2);}
/* --- sidebar --- */
.wb-side-brand{font-weight:600; font-size:1.2rem; letter-spacing:.01em; color:var(--ink);}
.wb-side-brand .dot{color:var(--accent);}
.wb-side-sub{text-transform:uppercase; letter-spacing:.1em; font-size:.68rem; color:var(--ink-3);
  font-weight:600; margin:.15rem 0 .2rem;}
.wb-side-label{text-transform:uppercase; letter-spacing:.09em; font-size:.7rem; color:var(--accent);
  font-weight:600; margin:1.1rem 0 .1rem;}
.wb-side-foot{font-size:.72rem; color:var(--ink-3); line-height:1.55; margin-top:1.2rem;
  border-top:1px solid var(--line); padding-top:.8rem;}
</style>
"""

HEADER_HTML = """
<div class="wb-head">
  <div class="wb-brand">WINGBEAT<span class="dot">.</span></div>
  <div class="wb-sub">Detección acústica de <i>Aedes aegypti</i> y su degradación bajo ruido
  urbano costarricense</div>
  <div class="wb-rule"></div>
</div>
"""

FOOTER_HTML = """
<div class="wb-foot">
<b>Nota.</b> Esta pantalla es una <i>ilustración</i> en vivo, no la medición formal
(<code>evaluate.py</code>, por segmento y a umbral 0.50). El ruido en vivo usa grabaciones CR
que ningún modelo vio en entrenamiento (tomas 01/02). A prevalencia real (~1&nbsp;%) la
precisión cae bastante más que aquí — ver <code>results/degradation_table.md</code> §8.
</div>
"""


@st.cache_resource(show_spinner=False)
def get_models() -> dict:
    # Load the models once and run one warm-up inference.
    models = inference.load_all_models()
    # One inference now builds the TF graph, so the first real prediction isn't slow.
    warm = np.random.default_rng(0).normal(0.0, 0.05, config.SAMPLE_RATE).astype(np.float32)
    for name, model in models.items():
        inference.predict_waveform(warm, model, inference.MODEL_KIND[name])
    return models


def sample_label(path: Path) -> str:
    # 'pos_aegypti_01.wav' -> 'Aedes aegypti - positivo - #1' (shown in the UI).
    stem = path.stem
    prefix = stem.split("_")[0]
    num = stem.split("_")[-1]
    slug = "_".join(stem.split("_")[1:-1])
    species = PRETTY_SPECIES.get(slug, slug)
    kind = "positivo" if prefix == "pos" else "negativo"
    return f"{species} · {kind} · #{int(num)}"


def _env_label(env: str) -> str:
    return ENV_LABELS.get(env, env)


def load_clip(source) -> np.ndarray | None:
    # Prepare the waveform and check it has content; None with a warning if not.
    try:
        wav = inference.prepare_waveform(source)
    except Exception as exc:  # noqa: BLE001 - archivo raro: avisar, no reventar
        st.error(f"No pude leer el audio: {exc}")
        return None
    if inference.count_segments(wav) == 0:
        st.warning("El audio es muy corto o está en silencio — probá otra muestra o un WAV más largo.")
        return None
    return wav


def _sparkline(probs: np.ndarray) -> str:
    # Small SVG sparkline of the per-window probability.
    if probs is None or len(probs) < 2:
        return ""
    w, h, n = 240.0, 32.0, len(probs)
    pts = " ".join(f"{i / (n - 1) * w:.1f},{h - float(np.clip(p, 0, 1)) * h:.1f}"
                   for i, p in enumerate(probs))
    y50 = h - 0.5 * h
    return (f'<svg class="wb-spark" viewBox="0 0 {w:.0f} {h:.0f}" preserveAspectRatio="none">'
            f'<line class="wb-spark-mid" x1="0" y1="{y50:.1f}" x2="{w:.0f}" y2="{y50:.1f}"/>'
            f'<polyline class="wb-spark-line" points="{pts}"/></svg>')


def _verdict_card(title: str, role: str, result) -> str:
    # HTML for one verdict card: model, chip, confidence and sparkline.
    if result is None or result.n_segments == 0:
        chip_cls, chip_txt = "neutral", "SIN DATO"
        body = '<div class="wb-meta">sin ventanas analizables</div>'
    else:
        detect = result.verdict == 1
        chip_cls = "detect" if detect else "fail"
        chip_txt = "DETECTA" if detect else "NO DETECTA"
        pct = float(np.clip(result.aggregate, 0.0, 1.0)) * 100.0
        body = (
            f'<div class="wb-conf-row"><span>P(aedes)</span>'
            f'<span class="wb-num">{result.aggregate:.3f}</span></div>'
            f'<div class="wb-bar"><div class="wb-bar-fill wb-bar-fill--{chip_cls}" '
            f'style="width:{pct:.1f}%"></div></div>'
            f'{_sparkline(result.segment_probs)}'
            f'<div class="wb-meta">{result.n_segments} ventana(s) · umbral 0.50</div>'
        )
    return (
        '<div class="wb-card">'
        f'<div class="wb-card-head"><span class="wb-model">{title}</span>'
        f'<span class="wb-role">{role}</span></div>'
        f'<div class="wb-chip wb-chip--{chip_cls}">{chip_txt}</div>'
        f'<div class="wb-conf">{body}</div>'
        '</div>'
    )


def render_verdict_triptych(results: dict) -> str:
    cards = "".join(_verdict_card(title, role, results.get(name))
                    for name, title, role in MODEL_COLUMNS)
    return f'<div class="wb-grid">{cards}</div>'


def section_header(idx: str, title: str, desc: str) -> None:
    st.markdown(
        f'<div class="wb-section"><span class="wb-idx">{idx}</span>'
        f'<span class="wb-title">{title}</span><span class="wb-desc">{desc}</span></div>',
        unsafe_allow_html=True,
    )


def render_spectrogram(waveform: np.ndarray):
    # Themed Mel spectrogram, with the wingbeat band marked.
    mel_db = mel_spectrogram(waveform)
    duration = len(waveform) / config.SAMPLE_RATE
    lo, hi = config.WINGBEAT_BAND_HZ
    mel_freqs = librosa.mel_frequencies(n_mels=config.N_MELS, fmin=config.FMIN, fmax=config.FMAX)
    bin_lo, bin_hi = np.searchsorted(mel_freqs, [lo, hi])

    fig, ax = plt.subplots(figsize=(4.4, 2.2))
    fig.patch.set_alpha(0.0)
    ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma",
             extent=(0.0, duration, 0.0, float(mel_db.shape[0])))
    # Wingbeat band: dashed guide lines (a teal fill gets lost over the magma colormap).
    for edge in (bin_lo, bin_hi):
        ax.axhline(edge, color=BAND, lw=0.9, ls=(0, (4, 3)), alpha=0.95)
    ax.text(duration * 0.985, bin_hi + 1.5, "aleteo 400–700 Hz", color=BAND, fontsize=6.5,
           ha="right", va="bottom")
    ax.set_xlabel("tiempo (s)", fontsize=7.5, color=INK_2)
    ax.set_ylabel("banda mel", fontsize=7.5, color=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(LINE)
    ax.tick_params(labelsize=6.5, colors=INK_3)
    fig.tight_layout(pad=0.4)
    return fig


def _randomize_selection() -> None:
    # Callback: pick a random sample, environment and level.
    samples = sorted(SAMPLES_DIR.glob("*.wav"))
    if samples:
        st.session_state["fuente_pick"] = "Muestra curada"
        st.session_state["sample_pick"] = sample_label(random.choice(samples))
    st.session_state["env_pick"] = random.choice(list(config.CR_NOISE_ENVIRONMENTS))
    st.session_state["level_pick"] = random.choice(list(NOISE_LEVELS))


def sidebar_controls() -> tuple[np.ndarray | None, str, str | None]:
    # Sidebar inputs. Returns (waveform, environment, condition key).
    with st.sidebar:
        st.markdown('<div class="wb-side-brand">WINGBEAT<span class="dot">.</span></div>'
                    '<div class="wb-side-sub">panel de control</div>', unsafe_allow_html=True)

        st.markdown('<div class="wb-side-label">Entrada</div>', unsafe_allow_html=True)
        fuente = st.radio("Fuente", ["Muestra curada", "Subir WAV"], key="fuente_pick",
                          label_visibility="collapsed")
        wav = None
        if fuente == "Muestra curada":
            samples = sorted(SAMPLES_DIR.glob("*.wav"))
            if not samples:
                st.error("No hay muestras. Corré `python -m scripts.make_demo_samples`.")
            else:
                labels = {sample_label(p): p for p in samples}
                if "sample_pick" not in st.session_state:
                    st.session_state["sample_pick"] = next(iter(labels))
                pick = st.selectbox("Muestra", list(labels), key="sample_pick",
                                    label_visibility="collapsed")
                wav = load_clip(labels[pick])
        else:
            up = st.file_uploader("WAV (mono/estéreo, se resamplea a 8 kHz)", type=["wav"])
            if up is not None:
                wav = load_clip(up)

        st.markdown('<div class="wb-side-label">Ruido de fondo CR</div>', unsafe_allow_html=True)
        if "env_pick" not in st.session_state:
            st.session_state["env_pick"] = config.CR_NOISE_ENVIRONMENTS[0]
        if "level_pick" not in st.session_state:
            st.session_state["level_pick"] = "Limpio"
        env = st.selectbox("Ambiente", list(config.CR_NOISE_ENVIRONMENTS),
                           format_func=_env_label, key="env_pick")
        level_label = st.select_slider("Nivel de ruido (SNR)", options=list(NOISE_LEVELS),
                                       key="level_pick")
        st.caption("Menor SNR = más ruido. A 0 dB el ruido es tan fuerte como el aleteo.")

        st.markdown('<div class="wb-side-label">Atajo</div>', unsafe_allow_html=True)
        st.button("Selección aleatoria", on_click=_randomize_selection, width="stretch")

        st.markdown(
            '<div class="wb-side-foot">Los modelos y las grabaciones se cargan del repositorio. '
            'Sin conexión a internet.</div>', unsafe_allow_html=True)

    return wav, env, NOISE_LEVELS[level_label]


def render_signal_panel(wav: np.ndarray, x: np.ndarray, env: str, cond: str | None) -> None:
    # Audio and spectrogram, original vs.
    level_txt = {"easy": "20 dB", "medium": "10 dB", "hard": "0 dB"}.get(str(cond), "")
    ca, cb = st.columns(2)
    with ca:
        st.markdown('<div class="wb-label">Original</div>', unsafe_allow_html=True)
        st.audio(wav, sample_rate=config.SAMPLE_RATE)
        st.pyplot(render_spectrogram(wav), width="stretch")
    with cb:
        if cond is None:
            st.markdown('<div class="wb-label">Con ruido CR</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="wb-empty" style="padding:2.6rem 1.2rem;">Subí el <b>nivel de ruido</b> '
                'en el panel de control para mezclar ruido costarricense real y ver la '
                'degradación.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="wb-label">Con ruido · {_env_label(env)} · {level_txt}</div>',
                        unsafe_allow_html=True)
            st.audio(x, sample_rate=config.SAMPLE_RATE)
            st.pyplot(render_spectrogram(x), width="stretch")
    st.caption("Las líneas punteadas marcan la banda del aleteo (400–700 Hz). Con ruido, esa banda "
               "se llena de energía ajena — por eso el modelo pierde la señal.")


def render_recordings() -> None:
    # The 15 CR noise recordings, grouped by environment.
    for cr_env in config.CR_NOISE_ENVIRONMENTS:
        with st.expander(_env_label(cr_env)):
            takes = sorted(config.NOISE_CR_DIR.glob(f"{cr_env}_*.wav"))
            for take_col, take_path in zip(st.columns(len(takes) or 1), takes):
                info = sf.info(str(take_path))
                take_no = take_path.stem.rsplit("_", 1)[-1]
                role = "reservada · entrena la cura" if take_no == "03" else "ruido en vivo"
                take_col.markdown(
                    f'<div class="wb-meta">toma {take_no} · {info.duration:.0f}s · {role}</div>',
                    unsafe_allow_html=True)
                take_col.audio(str(take_path))


def main() -> None:
    st.set_page_config(page_title="WINGBEAT", layout="wide", initial_sidebar_state="expanded")
    _use_plex_in_matplotlib()
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(HEADER_HTML, unsafe_allow_html=True)

    with st.spinner("Cargando modelos…"):
        models = get_models()
    if not models:
        st.error("No hay modelos en models/. Entrená el pipeline antes de correr la demo.")
        st.stop()

    wav, env, cond = sidebar_controls()

    if wav is None:
        st.markdown(
            '<div class="wb-empty">Elegí un audio en el <b>panel de control</b> '
            '(a la izquierda) para comenzar.</div>', unsafe_allow_html=True)
        st.markdown(FOOTER_HTML, unsafe_allow_html=True)
        return

    x = wav
    if cond is not None:
        try:
            x = inference.add_cr_noise(wav, env, config.SNR_LEVELS_DB[cond])
        except Exception as exc:  # noqa: BLE001
            st.error(f"No pude mezclar el ruido «{_env_label(env)}»: {exc}")
            st.stop()

    section_header("01", "Señal", "el audio que se analiza y su espectrograma")
    render_signal_panel(wav, x, env, cond)

    section_header("02", "Veredicto", "qué dice cada modelo sobre este clip")
    results = inference.predict_all(x, models)
    st.markdown(render_verdict_triptych(results), unsafe_allow_html=True)

    section_header("03", "Grabaciones de campo", "ruido CR real · 5 ambientes × 3 tomas de ~2 min")
    render_recordings()

    st.markdown(FOOTER_HTML, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
