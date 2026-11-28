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