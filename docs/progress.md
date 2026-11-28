# Progress log

Decisions, problems and next steps. Newest entries at the bottom.

---

## 2026-11-25 — Project set-up and plan

**Context**

- Goal: remove power-line interference from an ECG-like signal using a reference measurement
  of the interference, then test one justified change to the algorithm.
- Approach: small steps (data, baseline, modification, metrics, experiments, figures), each with
  its own tests.

**Decisions**

- Track B: adaptive ECG noise cancellation (Thakor & Zhu, 1991). Only LMS reference-noise
  cancellation is reproduced; arrhythmia detection is out of scope.
- Data: synthetic ECG-like signals from the course guide (250 Hz, 12 s, 50 Hz interference
  through an unknown 3-tap pathway).
- Modification: normalised LMS (NLMS), step size divided by the reference input energy.
  - Hypothesis: a fixed LMS step tuned at one reference amplitude is too slow or unstable at
    others; NLMS should be less sensitive to amplitude.
  - Might hurt: very weak references; at the tuning amplitude LMS may be as good.
- Split by whole recordings: development seeds 0–19, validation 100–109, test 1000–1019.
- Primary metric: output SNR (dB) after a 1 s warm-up; larger is better.

**Next steps**

 Implement the data generator with tests.



## 2026-11-25 — Data generator

**Done**

- `src/data.py`: `make_ecg(seed, seconds, fs, scale)` returns `(clean, measured, reference)`.
  It keeps the course guide's order of random draws, so a given seed gives exactly the guide's
  signals.
- `tests/test_data.py`: shapes and finite values, reproducibility, scale property, R-peak
  amplitude, invalid arguments (6 tests, all passing).

**Next steps**

 Implement the baseline LMS filter with tests.


---

## 2026-11-25 — Baseline LMS

**Done**

- `src/baseline.py`: `lms_filter(d, r, M, mu)` returns the cleaned output `e` and final weights
  `w`. Weights start at zero; the output is computed before each weight update.
- `tests/test_baseline.py`: two hand-computed examples, zero reference and zero step size,
  convergence to the true pathway, interference reduction on a synthetic ECG, shapes and
  no input mutation, invalid arguments (11 tests).
- Tagged `v0.1-baseline` before any modification.

**Decision**

- Update written as `w += mu * e * x` .
  Papers that differentiate the
  full squared error write `2 * mu`; the factor of two is absorbed into `mu`.

**Next steps**

 Implement the NLMS modification with tests.


---

## 2026-11-25 — NLMS modification

**Done**

- `src/modified.py`: `nlms_filter(d, r, M, mu, eps=1e-6, normalize=True, reference_energy=None)`.
  The only change from the baseline is the step: `mu / (eps + x @ x)` instead of `mu`.
- Ablation switch: `normalize=False` reproduces LMS exactly; with `reference_energy=E` the step is
  fixed to `mu / (eps + E)`, which keeps the NLMS step-size scale but removes the per-sample
  normalisation.
- `tests/test_modified.py`: hand example, scale invariance,
  ablation equivalences, zero reference, convergence, stability at a 100x reference where LMS
  diverges, shapes and no mutation, invalid arguments (14 tests).

**Next steps**

Evaluation metrics with tests.


---

## 2026-11-25 — Evaluation metrics

**Done**

- `src/metrics.py`: `mse`, `snr_db` (−inf for a diverged output), `r_peak_indices` and
  `peak_amplitude_error` (distortion at true R peaks), `paired_difference` (mean, sd, 95%
  Student-t interval, median, count of a > b).
- Uncertainty is computed across independent recordings (one value per seed), never across
  samples within a recording. Diverged pairs are excluded from the interval and counted.
- 13 new tests (44 in total).

**Next steps**

1. Experiment: validation-only parameter selection, test evaluation, configs and a runner script.


---

## 2026-11-28 — Experiment and final results

**Done**

- Stress options in the data generator: pathway change at a given time, and ECG leakage into the
  reference (`r <- r + leakage * s`). Neither draws extra random numbers.
- `src/experiment.py` and `run_experiments.py`, built in small tested steps: config checks,
  validation-only grid search, ablation energy, test evaluation, summaries, timing, environment.
- Ran `configs/final.json` on my laptop (Apple M5, Python 3.12.16); results in `results/final/`.

**Protocol (fixed in the configs before running the final test)**

- Tune at reference scale 1.0 on validation seeds 100–109, then freeze. Equal budgets:
  M in {2, 3, 4, 8, 16} and 8 step sizes for each method. eps = 1e-6 fixed.
- Ablation: NLMS (M, mu) with a fixed step mu / (eps + E), E = mean validation input energy.
- Test: 5 conditions (scale 0.3 / 1.0 / 3.0, pathway change at 6 s, leakage 0.2),
  20 recordings each (seeds 1000–1019). Seed splits are checked to be disjoint in code.

**Results**

- Selected: LMS M = 3, mu = 0.05; NLMS M = 3, mu = 0.1. Both chose the true pathway length.
- NLMS − LMS output SNR, mean [95% CI]: scale 0.3 −0.41 [−0.54, −0.29] dB; scale 1.0 −0.05
  [−0.18, +0.07] dB; scale 3.0 +11.24 [+10.90, +11.59] dB; pathway change +0.59 dB;
  leakage +0.47 dB. No run diverged.
- Ablation at scale 3.0 is 14.0 dB worse than NLMS, so the gain comes from the per-sample
  normalisation, not from a different step-size scale.
- Against my predictions: confirmed at scale 3.0 and at the tuning scale; at scale 0.3 NLMS is
  slightly worse, as predicted for weak references. It adapts faster (warm-up SNR 15.8 vs
  12.4 dB) but has a larger R-peak error (0.050 vs 0.021).
- Leakage lowers both methods by about 3.5 dB: the assumption that the reference contains no ECG
  is violated.
- Runtime per 12 s recording: LMS 2.96 ms, NLMS 4.03 ms (+36 %), ablation 2.93 ms.

**Next steps**

1. Figures generated from the saved results.
2. Complete the README. 


---

## 2026-11-28 — Figures, README and reproduction check

**Done**

- `make_figures.py` writes four figures to `results/final/figures/` from the saved results.
- README completed.

**Decisions and problems**

- Fig. 2 gives each condition its own x-axis, so the small effects are not hidden by the 11 dB one.
- Fig. 3 uses a causal moving average to avoid artefacts at the ends.
- Averaging the error over 251 beats aligned on R peaks revealed the NLMS distortion at scale 0.3,
  which a single error trace did not show.
- A lint check found formatting issues, to be fixed before submission.

**Next steps**

1. Write the report, record the video, clean up formatting and tag `v1.0-submission`.