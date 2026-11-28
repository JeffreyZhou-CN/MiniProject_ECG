"""Generate the report figures from a results folder written by run_experiments.py.

Usage (from the repository root):
    python make_figures.py --results results/final

Figures are written to <results>/figures/ as PNG files. Summary figures read only the
saved CSV files; no numbers are typed in by hand.
"""

# Summary figures read only the saved CSV files; no numbers are typed in by hand.

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

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
    figure_paired(results_dir, out_dir)
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()