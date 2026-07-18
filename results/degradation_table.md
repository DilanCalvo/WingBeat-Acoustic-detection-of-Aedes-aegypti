# Degradation table — WINGBEAT

All four conditions hold **exactly the same 6206 test segments**; the only thing that changes between them is the Costa Rican background noise mixed on top. Any difference between columns is caused by the noise.

| condition | SNR | what it is |
|---|---|---|
| `clean` | — | lab audio, no noise |
| `easy` | 20 dB | light background noise |
| `medium` | 10 dB | noise comparable to the signal |
| `hard` | 0 dB | noise as loud as the wingbeat |

## 1. Headline result

`FN rate` = of all the *Aedes* actually present, what fraction went **undetected**. `ROC-AUC` and `MCC` do not depend on the threshold.

> **Trivial floor = 0.651.** A classifier that answers "aedes" to everything gets that F1 (base rate 48.3 %). **Every F1 in this table has to be read against that number, not against zero.** `MCC` is the other safeguard: it is exactly **0.000** for that trivial classifier, so it cannot be inflated by predicting everything positive.

| model | condition | F1 [95% CI] | vs. trivial | precision | recall | **FN rate** | ROC-AUC | MCC |
|---|---|---|---|---|---|---|---|---|
| CNN | `clean` | **0.928** [0.917–0.937] | +0.276 | 0.932 | 0.923 | **0.077** | 0.983 | 0.861 |
| CNN | `easy` | **0.009** [0.005–0.015] | -0.642 | 0.933 | 0.005 | **0.995** | 0.820 | 0.044 |
| CNN | `medium` | **0.002** [0.000–0.005] | -0.649 | 1.000 | 0.001 | **0.999** | 0.709 | 0.023 |
| CNN | `hard` | **0.001** [0.000–0.002] | -0.650 | 1.000 | 0.000 | **1.000** | 0.611 | 0.013 |
| RandomForest | `clean` | **0.896** [0.878–0.910] | +0.244 | 0.848 | 0.948 | **0.052** | 0.961 | 0.792 |
| RandomForest | `easy` | **0.056** [0.045–0.068] | -0.595 | 0.870 | 0.029 | **0.971** | 0.848 | 0.099 |
| RandomForest | `medium` | **0.005** [0.001–0.008] | -0.647 | 0.778 | 0.002 | **0.998** | 0.816 | 0.022 |
| RandomForest | `hard` | **0.001** [0.000–0.002] | -0.650 | 1.000 | 0.000 | **1.000** | 0.774 | 0.013 |
| CNN + CR noise | `clean` | **0.917** [0.906–0.928] | +0.266 | 0.908 | 0.928 | **0.072** | 0.977 | 0.839 |
| CNN + CR noise | `easy` | **0.907** [0.895–0.918] | +0.256 | 0.908 | 0.907 | **0.093** | 0.972 | 0.821 |
| CNN + CR noise | `medium` | **0.886** [0.871–0.897] | +0.234 | 0.879 | 0.893 | **0.107** | 0.960 | 0.777 |
| CNN + CR noise | `hard` | **0.781** [0.763–0.795] | +0.129 | 0.845 | 0.725 | **0.275** | 0.900 | 0.610 |

## 2. Degradation by noise environment (CNN)

~1,241 segments per cell. Telling *which* noise breaks the detector would be a finding, but it has to be measured with the right metric.

> **Why environments are compared with ROC-AUC and not F1:** at the fixed 0.5 threshold F1 **collapses to ~0 in all five environments**, so there is no dynamic range left to tell them apart. ROC-AUC is **not saturated** and does show which one degrades most. Ranking environments by F1 here would be reading measurement noise.

| environment | AUC `clean` | AUC `easy` | AUC `medium` | AUC `hard` | AUC drop | F1 `clean`→`hard` |
|---|---|---|---|---|---|---|
| `kitchen_fan` | 0.983 | 0.858 | 0.737 | 0.617 | **0.367** | 0.920 → 0.000 |
| `park_green` | 0.981 | 0.822 | 0.749 | 0.656 | **0.324** | 0.923 → 0.000 |
| `street_traffic` | 0.983 | 0.825 | 0.710 | 0.604 | **0.379** | 0.928 → 0.000 |
| `tv_room` | 0.980 | 0.849 | 0.800 | 0.735 | **0.245** | 0.931 → 0.003 |
| `voices_market` | 0.986 | 0.787 | 0.682 | 0.627 | **0.359** | 0.936 → 0.000 |

## 3. Domain confound control

The models partly exploit a **channel cue** (they reject HumBugDB negatives more easily than Wingbeats ones). CR noise masks that cue, so **part of the measured degradation could be a dataset artifact dying rather than the wingbeat signal**. Control: repeat the measurement on the **same-domain** subset (Wingbeats only), which is free of the confound.

| condition | F1 full test | F1 same-domain (Wingbeats) | FP rate Wingbeats neg. | FP rate HumBugDB neg. | gap |
|---|---|---|---|---|---|
| `clean` | 0.928 | **0.928** | 0.118 | 0.000 | 0.118 |
| `easy` | 0.009 | **0.009** | 0.001 | 0.000 | 0.001 |
| `medium` | 0.002 | **0.002** | 0.000 | 0.000 | 0.000 |
| `hard` | 0.001 | **0.001** | 0.000 | 0.000 | 0.000 |

> If the gap **closes** once noise is added, that is direct evidence the channel cue existed and the noise erased it. If the same-domain F1 degrades just like the full one, the confound does **not** contaminate the headline result.

## 4. Ambient false alarms — *do not cry wolf at a passing car*

FP rate over the **418 negatives with no insect at all** (`humbugdb/background`): the system *should not* fire on pure ambient noise.

| condition | CNN | RandomForest | CNN + CR noise |
|---|---|---|---|
| `clean` | 0.000 | 0.000 | 0.007 |
| `easy` | 0.000 | 0.000 | 0.010 |
| `medium` | 0.000 | 0.000 | 0.007 |
| `hard` | 0.000 | 0.000 | 0.012 |

> **Careful reading this table: a 0.000 here is NOT a virtue.** Under noise the model stops predicting "aedes" **for almost everything** (recall goes to 0), so of course it raises no false alarms. It is not that it recognizes empty ambient noise well, it is that it no longer recognizes anything. This row only means something in `clean`.

> Not broken down by environment: that would be ~84 segments per cell, too fine to be reliable.

## 5. Diagnosis — noise destroying the signal, or a miscalibrated threshold?

**Read the last column against the trivial floor (0.651), not against zero.** An oracle F1 of 0.658 sounds like a partial rescue; next to the 0.651 you get by guessing, it is nothing.

| model | condition | F1 @ 0.5 | ROC-AUC | F1 at oracle threshold | vs. trivial | does the threshold rescue it? |
|---|---|---|---|---|---|---|
| CNN | `clean` | 0.928 | 0.983 | 0.931 | +0.280 | — |
| CNN | `easy` | 0.009 | 0.820 | 0.748 | +0.096 | barely |
| CNN | `medium` | 0.002 | 0.709 | 0.677 | +0.025 | **No** (~ guessing) |
| CNN | `hard` | 0.001 | 0.611 | 0.658 | +0.007 | **No** (~ guessing) |
| RandomForest | `clean` | 0.896 | 0.961 | 0.897 | +0.246 | — |
| RandomForest | `easy` | 0.056 | 0.848 | 0.800 | +0.149 | yes |
| RandomForest | `medium` | 0.005 | 0.816 | 0.777 | +0.126 | yes |
| RandomForest | `hard` | 0.001 | 0.774 | 0.746 | +0.095 | barely |
| CNN + CR noise | `clean` | 0.917 | 0.977 | 0.920 | +0.269 | — |
| CNN + CR noise | `easy` | 0.907 | 0.972 | 0.910 | +0.258 | yes |
| CNN + CR noise | `medium` | 0.886 | 0.960 | 0.889 | +0.238 | yes |
| CNN + CR noise | `hard` | 0.781 | 0.900 | 0.817 | +0.166 | yes |

> **Correction relative to an earlier version of this table.** It used to report the oracle F1 (0.748 / 0.677 / 0.658) as if it measured how much recalibrating recovers. **Without the trivial floor next to it that is misleading:** at 10 dB and 0 dB the oracle only beats a classifier answering "aedes" to everything by +0.026 and +0.007. The honest reading is that **moving the threshold does NOT rescue those two conditions**: there is no spare information to recalibrate. Only `easy` (20 dB) has a real margin (+0.097, AUC 0.820).

> The **oracle F1 picks the threshold while looking at the answers**: it is an *upper bound*, not deployable performance. It is only useful for the diagnosis: high AUC + low F1@0.5 + high oracle means **miscalibration** (fixable by moving the threshold). Low AUC + low oracle means **the noise destroyed the information** (no threshold saves it).

### Does the noise spectrum explain which environment hurts the detector most?

Prior hypothesis, stated to be tested rather than assumed: noise that **overlaps the wingbeat band (400-700 Hz)** should interfere more. It is tested against the **AUC** drop, not F1, because of the saturation explained in section 2.

| environment | % of noise energy in the band | AUC drop (clean->hard) |
|---|---|---|
| `kitchen_fan` | 10.4 % | 0.367 |
| `voices_market` | 8.3 % | 0.359 |
| `tv_room` | 5.5 % | 0.245 |
| `park_green` | 3.6 % | 0.324 |
| `street_traffic` | 2.8 % | 0.379 |

> Pearson correlation: **r = 0.17** (n = 5 environments, which is few enough that this is *suggestive, not conclusive*).
> **The spectral-overlap hypothesis does NOT hold with this data.** The detector does not break *especially* on noise that occupies its band: it breaks on **any** real ambient noise. Reported as such, against the initial hunch, because that is what the numbers say.

## 6. The cure — can it be fixed?

Two interventions, with very different costs:

1. **Recalibrate the threshold** (free, no retraining). The threshold is chosen on a **noisy validation set**: val segments + the `cal` half of the reserved noise. Neither those segments nor that noise audio are in the test set. *This is not the oracle threshold.*
2. **Retrain the CNN with CR noise** (`--augment`): one noisy copy of every training segment, at a random SNR between 0 and 20 dB, using **only** the `aug` half of the reserved noise. The test noise is still unseen by it.

| strategy | threshold | `clean` | `easy` 20 dB | `medium` 10 dB | `hard` 0 dB |
|---|---|---|---|---|---|
| CNN (as it was) | 0.50 | 0.928 | 0.009 | 0.002 | 0.001 |
| CNN **+ recalibrated threshold** | 0.000 | 0.661 | 0.672 | 0.667 | 0.654 |
| RandomForest (as it was) | 0.50 | 0.896 | 0.056 | 0.005 | 0.001 |
| RandomForest **+ recalibrated threshold** | 0.060 | 0.786 | 0.771 | 0.776 | 0.720 |
| **CNN retrained with CR noise** | 0.50 | 0.917 | 0.907 | 0.886 | 0.781 |
| **CNN with noise + recalibrated threshold** | 0.274 | 0.915 | 0.908 | 0.885 | 0.809 |
| *trivial floor (answer "aedes" to everything)* | — | *0.651* | *0.651* | *0.651* | *0.651* |

**The same table in MCC** (it is 0.000 for the trivial classifier, so a fake rescue cannot hide):

| strategy | `clean` | `easy` | `medium` | `hard` |
|---|---|---|---|---|
| CNN + recalibrated threshold | 0.143 | 0.212 | 0.176 | 0.097 |
| RandomForest + recalibrated threshold | 0.564 | 0.526 | 0.528 | 0.418 |
| **CNN retrained with CR noise** | 0.839 | 0.821 | 0.777 | 0.610 |

> **CNN** — threshold chosen on validation: `0.0000`. DEGENERATE: recalibrating just pushes it to answer 'aedes' to everything
> **RandomForest** — threshold chosen on validation: `0.0600`. REAL: recalibrating does rescue it

> **Honest reading.** Recalibrating is free but **it does not help the CNN**: its optimal threshold goes essentially to zero, meaning the "solution" is to answer *aedes* to everything, so F1 rises to the trivial floor while MCC stays on the floor. The **RandomForest is recalibratable**, which is the irony here: **the simple model, the one that lost in the lab, is the only one you can rescue without retraining.**

## 7. UFRGS holdout — a real capture-device change

333 segments recorded **with a phone**, never seen in training (only 49 positives, a caveat worth stating). This is the second transferability axis: not synthetic noise, but a different real microphone.

| model | F1 | recall | ROC-AUC |
|---|---|---|---|
| CNN | 0.000 | 0.000 | 0.873 |
| RandomForest | 0.000 | 0.000 | 0.810 |
| CNN + CR noise | 0.000 | 0.000 | 0.579 |

> **The most uncomfortable finding, and the most important one: the cure is SPECIFIC, not general.** Retraining with CR noise fixed CR noise dramatically (F1 0.001 -> 0.781 at 0 dB), but **made transfer to a real device worse**: ROC-AUC on UFRGS falls from **0.873** (naive CNN) to **0.579** (augmented CNN), a hair above chance (0.5).

> In other words, teaching it Costa Rican noise made it robust **to that noise**, not robust *in general*. It specialized. **A microphone change is a different problem from a background-noise change, and solving one does not solve the other.** A real deployment would have to attack both separately, and none of this would have been visible without holding UFRGS out.

## 8. At what prevalence do these numbers hold? (deployment caveat)

Everything above was measured on a **deliberately balanced** test set: 48.3 % of the segments contain an *Aedes* (a consequence of `TARGET_SEGMENTS_PER_CLASS`). **In a real deployment a mosquito is not present half the time**, it is much closer to 1 %.

That does not change recall or AUC (neither depends on prevalence), but it **sinks precision**, because false positives multiply against a much larger base of negatives:

> `precision(pi) = TPR*pi / (TPR*pi + FPR*(1-pi))`

Precision of the **CNN + CR noise** (same model, same threshold, same TPR/FPR; only how rare the mosquito is changes):

| condition | recall (TPR) | FP rate | precision @ pi=50% | @ pi=10% | @ pi=1% |
|---|---|---|---|---|---|
| `clean` | 0.928 | 0.088 | 0.913 | 0.539 | 0.096 |
| `easy` | 0.907 | 0.086 | 0.914 | 0.540 | 0.097 |
| `medium` | 0.893 | 0.115 | 0.886 | 0.463 | 0.073 |
| `hard` | 0.725 | 0.124 | 0.854 | 0.393 | 0.056 |

> **How to read this without misleading anyone.** An F1 of 0.907 is real *in the regime where it was measured*. If the same detector were hung in a yard where the mosquito shows up 1 % of the time, **most of its alarms would be false**, even though it hears just as well. That is not a flaw in the model: it is the arithmetic of imbalanced classes, and it happens to any detector. The engineering answer is to raise the threshold (giving up recall) or to require several detections over time before firing, **not** to report the balanced precision as if it were the field one.

> Stated here because leaving it out would be the same kind of silence as omitting the trivial floor (section 1): a correct number presented in a context that makes it look like something else.

## 9. Observations

*(derived mechanically from the numbers above; the causal interpretation is written separately, from the data, with no mechanism assumed up front)*

- **CNN**: F1 goes from 0.928 (`clean`) to 0.001 (`hard`, 0 dB), a drop of **0.927**. The false-negative rate rises from 0.077 to 1.000, and ROC-AUC from 0.983 to 0.611.
- **RandomForest**: F1 goes from 0.896 (`clean`) to 0.001 (`hard`, 0 dB), a drop of **0.895**. The false-negative rate rises from 0.052 to 1.000, and ROC-AUC from 0.961 to 0.774.
- **CNN + CR noise**: F1 goes from 0.917 (`clean`) to 0.781 (`hard`, 0 dB), a drop of **0.137**. The false-negative rate rises from 0.072 to 0.275, and ROC-AUC from 0.977 to 0.900.

- **The drop at `easy` (20 dB) is mostly CALIBRATION, not blindness.** At the fixed 0.5 threshold the CNN gives F1 = 0.009 (recall 0.005, so it detects almost no mosquitoes), but its ROC-AUC is still **0.820** and, with a recalibrated threshold, it would reach F1 = **0.748**, which is **+0.096** over the trivial floor. The wingbeat information **is still there**; what broke is the decision point, calibrated for lab silence. Same pattern as the UFRGS holdout.
- **But at the loudest noise (`hard`, 0 dB) there is nothing left to recalibrate.** AUC falls to 0.611 and the oracle threshold only reaches F1 = 0.658, i.e. **+0.007** over the trivial floor (0.651): statistically indistinguishable from answering "aedes" to everything. Here the information really was destroyed, and **no threshold saves the detector, only retraining it with noise can.**
- **The simple model resists noise better.** At `hard` the RandomForest AUC (0.774) beats the CNN one (0.611), even though the CNN wins on clean audio (0.928 vs 0.896 F1). Winning in the lab and winning under noise **are not the same contest**.

- **The cure works, and the effect is large.** Retraining the CNN with CR noise (`--augment`) takes it, at `hard` (0 dB) and **at the default 0.5 threshold**, from F1 0.001 to **0.781**, +0.129 over the trivial floor, with MCC 0.610. And it **costs almost nothing on clean audio**: 0.917 vs 0.928 for the naive one. The test noise was unseen by it (recordings `_01`/`_02`).
- By environment (measured in AUC, see section 2): the one that degrades the CNN **most** is `street_traffic` (drop 0.379); the one that degrades it **least** is `tv_room` (drop 0.245). The spread between worst and best is only **0.134** of AUC: **no environment is safe**, the collapse comes from real noise being present at all, not from one particular kind of noise.

### Integrity check

The `clean` condition was re-extracted from disk through the **same path** as the noisy ones (manifest -> WAV -> features -> model). Its F1 has to reproduce the one from training; if it does not, the wiring is broken and this whole table would be invalid.

- CNN: clean F1 = 0.9275 vs 0.9275 at training time (diff 0.0000) OK
- RandomForest: clean F1 = 0.8955 vs 0.8956 at training time (diff 0.0001) OK
- CNN + CR noise: clean F1 = 0.9175 vs 0.9175 at training time (diff 0.0000) OK

---
*Generated by `src/evaluate.py` · decision threshold 0.5 · 95% CI by grouped bootstrap (1000 resamples of recordings).*