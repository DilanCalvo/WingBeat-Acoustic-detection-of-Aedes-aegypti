# WINGBEAT

Acoustic detection of *Aedes aegypti*, and a measurement of how much that detector
degrades when you put real Costa Rican street noise on top of it.

## What this is

Classifying mosquitoes by the sound of their wingbeat isn't new. It's been published
since 2018, with reported accuracies above 95%. What those papers usually skip is
what happens outside the lab: they train and test on clean recordings.

So this project trains two classifiers, records five real background-noise
environments around Costa Rica, mixes that noise into the test audio at three
controlled levels, and measures where the detector stops working. Then it tries to
fix it, and reports how far the fix actually goes.

Dengue surveillance here is slow and reactive. It depends on trapping mosquitoes and
having someone identify them under a microscope, which means the outbreak is often
already underway by the time the vector is confirmed. A cheap acoustic early warning
would help, but only if it survives contact with a noisy street, and that's the part
nobody had measured for this context.

## Results

Read the F1 scores against **0.651**, not against zero. That's what you get by
answering "aedes" to everything (the test set is 48.3% positive), so anything near
0.651 isn't really detecting anything.

F1 at a fixed 0.50 threshold. All four conditions use the same 6,206 test segments;
only the noise level changes.

| Model | clean | easy (20 dB) | medium (10 dB) | hard (0 dB) |
|---|---|---|---|---|
| CNN (lab-trained) | 0.928 | 0.009 | 0.002 | 0.001 |
| RandomForest (baseline) | 0.896 | 0.056 | 0.005 | 0.001 |
| **CNN + CR noise (retrained)** | 0.917 | **0.907** | **0.886** | **0.781** |
| *trivial floor* | *0.651* | *0.651* | *0.651* | *0.651* |

Three things came out of this:

**The lab-trained detector doesn't survive real noise.** At 20 dB SNR, which is noise
roughly ten times quieter than the mosquito, the CNN's F1 drops from 0.928 to 0.009
and its false-negative rate hits 0.995. It misses 99.5% of the mosquitoes that are
actually there.

**Retraining with Costa Rican noise fixes it.** The same architecture, retrained with
CR noise mixed into its own training set, holds F1 = 0.781 even at 0 dB (noise as
loud as the mosquito), and barely loses anything on clean audio (0.917 vs 0.928).

**But the fix is specific, not general.** That retrained model transfers *worse* to a
different microphone: its ROC-AUC on an unseen smartphone-recorded holdout (UFRGS)
drops from 0.873 to 0.579, which is close to a coin flip. Teaching a model to ignore
one kind of noise doesn't make it robust in general.

Full numbers, with confidence intervals, ROC-AUC, MCC, a per-environment breakdown
and a real-prevalence correction, are in
[`results/degradation_table.md`](results/degradation_table.md).

## How it works

The whole pipeline runs on a laptop CPU in minutes. No GPU, no cloud.

1. **Ingest.** Three public bioacoustics datasets, all natively 8 kHz mono, balanced
   to 30,000 segments (15,000 *Aedes* / 15,000 non-*Aedes*).
2. **Preprocess.** Mono, 8 kHz, volume-normalized, sliced into 1-second windows with
   50% overlap. Near-silent windows are dropped.
3. **Features.** Log-Mel spectrograms (64 mel bins) for the CNN, and 120-dimensional
   MFCC summary vectors (mean/std of the MFCCs plus their first and second
   derivatives) for the baseline. Both use identical STFT settings so the two models
   are compared fairly.
4. **Two models.** A RandomForest baseline (300 trees, balanced class weights) and a
   small CNN (three Conv-BatchNorm-Pool blocks, then global average pooling and a
   dense layer).
5. **Hybrid noisy test set.** Real CR field noise mixed onto **both classes** at three
   SNR levels (20/10/0 dB) with `audiomentations`. Noising only the positives would
   teach the model to detect noise instead of the mosquito, so every condition holds
   the same segments and only the noise level changes.
6. **Evaluate, then mitigate.** Measure the degradation, work out whether the signal
   was destroyed or just the threshold drifted, then retrain the CNN on a reserved,
   disjoint half of the noise recordings and measure again.

A few things that keep the numbers trustworthy: the train/val/test split is grouped
so a recording never lands on both sides, UFRGS is held out entirely as an
independent capture-device test, confidence intervals come from a grouped bootstrap
that resamples recordings rather than segments, and `tests/` pins the properties that
would otherwise break quietly (the realized SNR of each mix, the disjointness of the
noise pools, and the spectrogram normalization the CNN depends on).

## Running it

```bash
python -m venv venv
venv/Scripts/activate            # Windows;  source venv/bin/activate on Unix
pip install -r requirements.txt
```

The trained models are committed under `models/`, so the demo works straight away and
runs fully offline:

```bash
python -m streamlit run demo/app.py
```

To rebuild everything from scratch (each step writes what the next one reads):

```bash
python -m src.preprocess --all
python -m src.features
python -m src.train_baseline
python -m src.train_cnn --epochs 40
python -m src.build_hybrid_testset
python -m src.evaluate
# mitigation arm:
python -m src.build_hybrid_testset --split val
python -m src.build_noise_augmentation
python -m src.features --augmented
python -m src.train_cnn --augment --epochs 40
python -m src.calibrate
python -m src.evaluate --model all
# tests:
python -m pytest tests/ -q
```

HumBugDB ships as a flat folder plus a metadata CSV, so
`scripts/prepare_humbugdb.py` reorganizes it into the `data/raw/<dataset>/<class>/`
layout the pipeline expects. Run it once after downloading that dataset.

## Data

| Dataset | Role here | Link |
|---|---|---|
| Wingbeats | all *Aedes aegypti* positives, plus some negatives | [kaggle.com/potamitis/wingbeats](https://www.kaggle.com/potamitis/wingbeats) |
| HumBugDB | negatives only (other species + ambient) | [zenodo.org/record/4904800](https://zenodo.org/record/4904800) |
| UFRGS | smartphone holdout, device-shift test only | [github.com/ComputerNetworks-UFRGS](https://github.com/ComputerNetworks-UFRGS/Aedes_Detector_Smartphone_App_Paper) |

Every *Aedes* positive comes from Wingbeats, so the degradation measured here is
specific to Wingbeats positives mixed with CR noise. The five noise environments
(street traffic, kitchen fan, TV room, park, market voices) were recorded by us,
but **aren't versioned in this repository** — they were pulled out of the git
history and are no longer published here. To reproduce the noise-degradation
pipeline (`build_hybrid_testset.py`, `--augment` training, `calibrate.py`, the
demo's noise slider) you need your own `data/ruido_costarica/*.wav` (5
environments × 3 takes, 8 kHz mono, ~2 min each) sitting in that folder locally —
either your own field recordings or a copy of ours obtained separately.

## Limitations

- All positives come from one dataset, so the result is tied to that source.
- The UFRGS holdout only has 49 positives, and they come from a single recording. It's
  a warning sign, not a precise measurement.
- The noise fix doesn't carry over to a different microphone.
- The test set is balanced (48.3% positive) by design. At a realistic field prevalence
  of around 1%, most alarms would be false positives. That's the arithmetic of
  imbalanced classes, not a flaw in the model.
- Deliberately out of scope: no hardware device, no mobile app, no real-time
  inference, no epidemiological map, no free-flight field validation, and no
  male/female discrimination.

## Layout

```
src/                  pipeline: preprocess, features, split, train, evaluate, calibrate, inference
demo/                 offline Streamlit app (three models, live noise slider)
tests/                invariant + demo-inference tests
scripts/              one-off helpers (HumBugDB ingest, demo sample curation)
models/               committed models (RandomForest + two CNNs)
results/              degradation table, metrics JSONs, plots
data/ruido_costarica/ the five CR noise environments, kept locally only (not versioned, see "Data")
```

## License

Code released under the [MIT License](LICENSE). The datasets listed above are not
covered by it: each keeps its own upstream license, so check those before reusing the
data. The Costa Rican noise recordings are ours; they aren't published in this
repository, but if you obtain a copy separately they fall under the same MIT terms.
