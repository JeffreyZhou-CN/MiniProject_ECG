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


## Repository layout

| Path | Contents |
|---|---|
| [`src/data.py`](src/data.py) | Synthetic ECG, reference and interference generator (course guide), with stress options |
| [`src/baseline.py`](src/baseline.py) | Baseline LMS canceller |
| [`src/modified.py`](src/modified.py) | Modified NLMS canceller, with an ablation switch |
| [`src/metrics.py`](src/metrics.py) | MSE, SNR, R-peak error, paired differences with 95% intervals |
| [`src/experiment.py`](src/experiment.py) | Running methods, validation-only parameter selection, evaluation, timing |
| [`run_experiments.py`](run_experiments.py) | Runs the whole comparison from a configuration file and saves the results |
| [`make_figures.py`](make_figures.py) | Draws the figures from a saved results folder |
| [`configs/`](configs/) | Experiment settings: `smoke.json` (quick check) and `final.json` (full experiment) |
| [`tests/`](tests/) | Local checks for every module |
| [`results/final/`](results/final/) | Submitted results: CSV/JSON tables and `figures/` |
| [`docs/`](docs/) | Progress log, AI usage log and project conventions |

## Tested environment

| | |
|---|---|
| Machine | MacBook Air, Apple M5 (arm64) |
| Operating system | macOS 27.0.1 |
| Environment | conda, Python 3.12.16 |
| Packages | numpy 2.5.5, scipy 1.18.1, matplotlib 3.11.2, pytest 9.2.0 (pinned in [`requirements.txt`](requirements.txt)) |

## Installation

```bash
git clone https://github.com/JeffreyZhou-CN/MiniProject_ECG.git
cd MiniProject_ECG
conda env create -f environment.yml
conda activate ecg-anc
```

[`environment.yml`](environment.yml) installs Python 3.12 from conda-forge and the pinned packages from
`requirements.txt` with pip. 

All commands below are run from the repository root with the environment activated.

## Smoke run

```bash
python run_experiments.py --config configs/smoke.json
python make_figures.py --results results/smoke
```

A small grid with 3 validation and 3 test recordings; it checks the installation and shows all three
methods, but it is **not** the scientific evaluation. It takes a few seconds. The console prints the
selected parameters, the NLMS − LMS difference in output SNR for each condition and the runtimes.
Outputs are written to `results/smoke/`.

## Running the checks

```bash
python -m pytest
```

Expected: `73 passed` in under a minute. The tests cover hand-computed examples, edge cases and
properties of every module (for example: the LMS example from the course guide, scale invariance of
NLMS, convergence to the true interference pathway, and the disjoint seed splits).

## Reproducing the final results

```bash
python run_experiments.py --config configs/final.json
python make_figures.py --results results/final
```

The first command takes about 5 seconds and the second about 10 seconds on the tested machine. They
write to `results/final/` (folders are created automatically):

| File | Contents |
|---|---|
| `selected_params.json` | Parameters chosen on validation data, and the ablation's fixed energy |
| `validation_grid.csv` | Mean validation SNR for every grid point of both methods |
| `cases.csv` | One row per condition × test recording × method, with all metrics |
| `summary.csv` | Mean metrics per condition and method, with divergence counts |
| `paired.csv` | Paired differences (NLMS − LMS, NLMS − ablation) with 95% intervals |
| `timing.csv` | Median runtime per recording after a warm-up, with quartiles |
| `environment.json`, `config_used.json` | Machine and package versions; the exact settings used |
| `figures/fig1_example.png` … `fig4_failure_cases.png` | The four figures |

All metrics are deterministic: the same configuration gives the same values on any machine up to
floating-point rounding. Runtimes depend on the machine, so re-running overwrites `timing.csv` with
slightly different times; the submitted values are those in this repository.