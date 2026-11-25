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