# Adaptive ECG Noise Cancellation: LMS vs. Normalised LMS

EEP55C36 Computational Methods — Final Project, Track B (Biomedical).

This project reproduces the reference-noise cancellation method from:

> N. V. Thakor and Y.-S. Zhu, "Applications of Adaptive Filtering to ECG Analysis: Noise
> Cancellation and Arrhythmia Detection," IEEE Transactions on Biomedical Engineering,
> 38(8), 785–794, 1991. https://doi.org/10.1109/10.83591

A least-mean-squares (LMS) adaptive filter uses a reference measurement of the interference to
estimate and subtract power-line noise from a measured ECG-like signal. Only the noise-cancellation
part of the paper is reproduced; arrhythmia detection is out of scope.

The modification normalises the LMS step size by the energy of the reference input vector
(normalised LMS, NLMS). This is a known method, implemented independently here, not a new
discovery. The two methods are compared on the same inputs using reproducible experiments.

All data are synthetic; no clinical claims are made.