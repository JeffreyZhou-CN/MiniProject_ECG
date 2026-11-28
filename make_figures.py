"""Generate the report figures from a results folder written by run_experiments.py.

Usage (from the repository root):
    python make_figures.py --results results/final

Figures are written to <results>/figures/ as PNG files. Summary figures read only the
saved CSV files; no numbers are typed in by hand.
"""

# Summary figures read only the saved CSV files; no numbers are typed in by hand.

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.data import make_ecg  # noqa: E402
from src.experiment import expand_seeds, run_method  # noqa: E402
from src.metrics import snr_db  # noqa: E402

ROOT = Path(__file__).resolve().parent

COLOURS = {"lms": "#257de0", "nlms": "#f47230", "nlms_ablation": "#00b894"}
LABELS = {"lms": "LMS (baseline)", "nlms": "NLMS (modified)",
          "nlms_ablation": "Ablation (NLMS, normalisation off)"}
CONDITION_LABELS = {"scale_0.3": "Reference scale 0.3",
                    "scale_1.0": "Reference scale 1.0 (tuning scale)",
                    "scale_3.0": "Reference scale 3.0",
                    "pathway_change": "Pathway change at 6 s (stress)",
                    "leakage": "ECG leaks into reference (stress)"}
# Each method has one fixed colour and name in every figure; each condition has a
# readable title.


def read_csv(path):
    """Read a CSV file into a list of dicts (all values as strings)."""
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))

def load_run(results_dir):
    """Read the configuration and the selected parameters saved with the results."""
    
    config = json.loads((results_dir / "config_used.json").read_text())
    params = json.loads((results_dir / "selected_params.json").read_text())
    return config, params

def generate(config, seed, condition):
    """Regenerate one recording exactly as in the experiment, for a condition from the config."""
    
    return make_ecg(seed, seconds=config["seconds"], fs=config["fs"],
                    scale=condition.get("scale", 1.0),
                    pathway_after=condition.get("pathway_after"),
                    change_time=condition.get("change_time"),
                    leakage=condition.get("leakage", 0.0))


def error_power(clean, estimate, fs, smooth_seconds=0.2):
    """Causal moving average of the squared error (e - s)^2 over the last smooth_seconds.

    Returns
    -------
    ndarray, shape (N,) (Sample n averages samples max(0, n-k+1) ... n, k = smooth_seconds * fs.)
    """

    k = max(1, int(smooth_seconds * fs))
    squared = (np.asarray(estimate) - np.asarray(clean)) ** 2
    cumulative = np.concatenate([[0.0], np.cumsum(squared)])
    n = np.arange(1, len(squared) + 1)
    start = np.maximum(n - k, 0)
    return (cumulative[n] - cumulative[start]) / (n - start)
    # Window sums from a cumulative sum: the sum of samples a..b-1 is cumsum[b] - cumsum[a],
    # much faster than summing each window separately.

def figure_paired(results_dir, out_dir):
    """Figure 2: per-recording paired differences in output SNR, one panel per condition."""
    # Figure 2: one panel per condition; each dot is one recording's paired difference;
    # black marks the mean and 95% confidence interval.
    cases = read_csv(results_dir / "cases.csv")
    paired = read_csv(results_dir / "paired.csv")
    conditions = list(dict.fromkeys(row["condition"] for row in paired))
    # Take the condition names in the order they appear in the CSV, without duplicates.
    jitter = np.random.default_rng(0)
    # Random numbers with a fixed seed, used only to spread the dots vertically; their
    # horizontal positions are the real values.

    fig, axes = plt.subplots(len(conditions), 1, figsize=(7.0, 1.3 * len(conditions) + 1.0))
    for ax, name in zip(axes, conditions):
        for y, other in enumerate(("lms", "nlms_ablation")):
            snr = {m: {int(r["seed"]): float(r["snr_out_db"]) for r in cases
                       if r["condition"] == name and r["method"] == m}
                   for m in ("nlms", other)}
            diff = np.array([snr["nlms"][seed] - snr[other][seed] for seed in sorted(snr["nlms"])])
            ax.scatter(diff, y + jitter.uniform(-0.12, 0.12, diff.size), s=14,
                       color=COLOURS[other], alpha=0.6, linewidths=0)
            # [中文] 每段记录一个点：同一个 seed 上 NLMS 的 SNR 减去另一种方法的 SNR。
            # One dot per recording: NLMS SNR minus the other method's SNR on the same seed.

            stats = next(r for r in paired if r["condition"] == name and r["metric"] == "snr_out_db"
                         and r["comparison"] == f"nlms - {other}")
            mean, low, high = (float(stats[k]) for k in ("mean", "ci_low", "ci_high"))
            ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o", color="black",
                        capsize=3, zorder=3)
            ax.annotate(f"{mean:+.2f} dB [{low:+.2f}, {high:+.2f}]", (0.995, y),
                        xycoords=("axes fraction", "data"), ha="right", va="center", fontsize=7.5)
            # Mean and interval come from paired.csv; the label sits in the empty right
            # margin so it does not cover the dots.

        ax.axvline(0, color="grey", lw=0.9)
        ax.set_yticks([0, 1], ["NLMS − LMS", "NLMS − ablation"])
        ax.set_ylim(-0.5, 1.5)
        ax.invert_yaxis()
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(lo, -0.05 * (hi - lo)), hi + 0.6 * (hi - lo))
        ax.set_title(CONDITION_LABELS.get(name, name), loc="left", fontsize=10)
        # A vertical line marks 0; extra room on the right holds the labels; each panel
        # has its own horizontal scale.

    axes[-1].set_xlabel("Difference in output SNR (dB); positive means NLMS is better")
    fig.suptitle("Paired comparison over 20 test recordings per condition (synthetic data)",
                 x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.005, "Dots: one recording each (vertical jitter only). Black: mean with "
             "95% Student-t interval. Each panel has its own x-axis scale.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_dir / "fig2_paired_snr.png", dpi=150)
    plt.close(fig)
    # Add the axis label, title and caption, save as a 150 dpi PNG, then close the figure
    # to free memory.

def figure_example(results_dir, out_dir):
    """Figure 1: input, LMS output and NLMS output for the first test seed at scale 3.0."""

    # Figure 1: one recording (first test seed, scale 3.0): input, LMS output and NLMS
    # output, each compared with the clean ECG.
    config, params = load_run(results_dir)
    fs, seconds, eps = config["fs"], config["seconds"], config["eps"]
    warmup = int(round(config["warmup_seconds"] * fs))
    seed = expand_seeds(config["test"]["seeds"])[0]
    clean, measured, reference = make_ecg(seed, seconds=seconds, fs=fs, scale=3.0)
    outputs = {m: run_method(m, measured, reference, params[m], eps) for m in ("lms", "nlms")}
   
    # Regenerate this recording with the saved configuration and run the filters with
    # the selected parameters through the same run_method used in the experiment.

    t = np.arange(len(clean)) / fs
    window = (t >= 8.0) & (t < 10.0)
    panels = [("Measured signal d (input)", measured, "grey"),
              ("LMS (baseline) output e", outputs["lms"], COLOURS["lms"]),
              ("NLMS (modified) output e", outputs["nlms"], COLOURS["nlms"])]

    # Show only 8-10 s (about two beats) so the details are visible.

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.6), sharex=True, sharey=True)
    for ax, (title, signal, colour) in zip(axes, panels):
        ax.plot(t[window], signal[window], color=colour, lw=1.0, label="signal named in title")
        ax.plot(t[window], clean[window], color="black", lw=0.9, ls="--",
                label="clean ECG s (ground truth)")
        ax.set_title(f"{title}: SNR {snr_db(clean, signal, start=warmup):.1f} dB "
                     "(whole recording after warm-up)", loc="left", fontsize=10)
        ax.set_ylabel("amplitude (a.u.)")

    # In each panel the coloured line is the signal and the black dashed line is the clean
    # ECG; the title gives the output SNR of this recording.
    # sharex/sharey give all three panels the same axis scales, so they compare directly.

    axes[-1].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("time (s); 8-10 s of the 12 s recording shown")
    fig.suptitle(f"Representative recording: test seed {seed}, reference scale 3.0 (synthetic data)",
                 x=0.01, ha="left", fontsize=11)
    fig.text(0.01, 0.005, f"Parameters chosen on validation data at scale 1.0: "
             f"LMS M={params['lms']['M']}, mu={params['lms']['mu']:g}; "
             f"NLMS M={params['nlms']['M']}, mu={params['nlms']['mu']:g}.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_dir / "fig1_example.png", dpi=150)
    plt.close(fig)

    # The caption states that the parameters were chosen on validation data at scale 1.0;
    # then save and close.

def figure_adaptation(results_dir, out_dir):
    """Figure 3: mean error power over time for scale 0.3, scale 3.0 and the pathway change."""
    config, params = load_run(results_dir)
    fs, eps = config["fs"], config["eps"]
    seeds = expand_seeds(config["test"]["seeds"])
    t = np.arange(int(config["seconds"] * fs)) / fs
    names = ["scale_0.3", "scale_3.0", "pathway_change"]

    fig, axes = plt.subplots(len(names), 1, figsize=(7.0, 6.4), sharex=True)
    for ax, name in zip(axes, names):
        condition = next(c for c in config["test"]["conditions"] if c["name"] == name)
        for method in ("lms", "nlms", "nlms_ablation"):
            curves = []
            for seed in seeds:
                clean, measured, reference = generate(config, seed, condition)
                e = run_method(method, measured, reference, params[method], eps)
                curves.append(error_power(clean, e, fs))
            mean_db = 10 * np.log10(np.maximum(np.mean(curves, axis=0), 1e-12))
            ax.plot(t, mean_db, color=COLOURS[method], lw=1.3, label=LABELS[method])
            # For each method: compute the error-power curve on every test recording,
            # average them, convert to dB and plot.
            # np.maximum(..., 1e-12) avoids taking the log of zero.
        ax.axvspan(0, config["warmup_seconds"], color="#eeeeee", zorder=0,
                   label="warm-up (not scored)")
        if condition.get("change_time") is not None:
            ax.axvline(condition["change_time"], color="grey", ls="--", lw=0.9)
            ax.annotate("pathway changes", (condition["change_time"], 0.9),
                        xycoords=("data", "axes fraction"), xytext=(-4, 0),
                        textcoords="offset points", ha="right", fontsize=7.5, color="grey")
    
        # A grey band marks the warm-up; a dashed line marks the moment of a pathway change.
        ax.set_title(CONDITION_LABELS[name], loc="left", fontsize=10)
        ax.set_ylabel("error power (dB)")

    axes[0].legend(loc="upper right", fontsize=7.5, ncol=2)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Adaptation: error power (e - s)^2, 0.2 s causal moving average, "
                 f"mean of {len(seeds)} test recordings", x=0.01, ha="left", fontsize=10)
    fig.text(0.01, 0.005, "Lower is better. 0 dB corresponds to an error power of 1 (a.u.)^2.",
             fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out_dir / "fig3_adaptation.png", dpi=150)
    plt.close(fig)

def main():
    """Command-line entry point."""
    # Command-line entry point: read a results folder and save the figures in its
    # figures subfolder.
    parser = argparse.ArgumentParser(description="Generate figures from saved results.")
    parser.add_argument("--results", default="results/final", help="results folder")
    args = parser.parse_args()
    results_dir = ROOT / args.results
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_example(results_dir, out_dir)
    figure_paired(results_dir, out_dir)
    figure_adaptation(results_dir, out_dir)
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()