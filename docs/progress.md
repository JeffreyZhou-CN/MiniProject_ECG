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

1. Implement the data generator with tests.



## 2026-11-25 — Data generator

**Done**

- `src/data.py`: `make_ecg(seed, seconds, fs, scale)` returns `(clean, measured, reference)`.
  It keeps the course guide's order of random draws, so a given seed gives exactly the guide's
  signals.
- `tests/test_data.py`: shapes and finite values, reproducibility, scale property, R-peak
  amplitude, invalid arguments (6 tests, all passing).

**Next steps**

1. Implement the baseline LMS filter with tests.


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

1. Implement the NLMS modification with tests.