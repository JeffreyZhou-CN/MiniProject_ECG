"""Run the comparison of LMS, NLMS and the ablation on synthetic ECG recordings.

Usage :
    python run_experiments.py --config configs/smoke.json    # quick check
    python run_experiments.py --config configs/final.json    # full experiment
"""
# Run the comparison of LMS, NLMS and the ablation. Run from the repository root and
# choose the configuration file with --config.

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy

from src.data import make_ecg
from src.experiment import (METHODS, check_disjoint, evaluate_case, expand_seeds, mean_input_energy, select_parameters, time_method)
from src.metrics import paired_difference


ROOT = Path(__file__).resolve().parent
# Absolute path of the repository root. Output paths start from here, so the script
# works whichever directory it is run from.


def write_csv(path, rows):
    """Write a list of dicts to a CSV file, with the keys of the first row as columns."""
    # Write a list of dicts to a CSV file, using the first dict's keys as column names.
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_config(config):
    """Check the fairness rules before running: disjoint seed splits and equal grid sizes."""
    # Check two fairness rules before running: the seed groups do not overlap, and both
    # methods have the same number of candidate step sizes.
    check_disjoint(expand_seeds(config["development_seeds"]),
                   expand_seeds(config["validation"]["seeds"]),
                   expand_seeds(config["test"]["seeds"]))
    if len(config["grid"]["lms_mu"]) != len(config["grid"]["nlms_mu"]):
        raise ValueError("LMS and NLMS step grids must have the same size (equal tuning budget)")


def load_config(path):
    """Read a JSON configuration file and validate it."""
    config = json.loads(Path(path).read_text())
    validate_config(config)
    return config


def select_all_parameters(config):
    """Select (M, mu) for LMS and NLMS on validation data; derive the ablation's energy.

    Returns
    -------
    selected : dict (Parameters for "lms", "nlms" and "nlms_ablation".)
    table : list of dict (Mean validation SNR for every grid point of both methods.)
    """
    # Select parameters for LMS and NLMS on the validation set; the ablation reuses the
    # NLMS M and mu plus the mean validation energy E, with no extra tuning.
    fs, seconds, eps = config["fs"], config["seconds"], config["eps"]
    warmup = int(round(config["warmup_seconds"] * fs))
    recordings = [make_ecg(seed, seconds=seconds, fs=fs, scale=config["validation"]["scale"])
                  for seed in expand_seeds(config["validation"]["seeds"])]
    selected, table = {}, []
    for method, steps in (("lms", config["grid"]["lms_mu"]), ("nlms", config["grid"]["nlms_mu"])):
        best, rows = select_parameters(method, config["grid"]["M"], steps, recordings, warmup, eps)
        selected[method] = best
        table.extend(rows)
    energy = mean_input_energy([reference for _, _, reference in recordings], selected["nlms"]["M"])
    selected["nlms_ablation"] = {"M": selected["nlms"]["M"], "mu": selected["nlms"]["mu"],
                                 "energy": energy}
    return selected, table

def evaluate_test_cases(config, params):
    """Run every method on every test recording of every condition, with frozen parameters.

    Returns
    -------
    list of dict (One row per condition x seed x method, with all metrics.)
    """

    fs, seconds, eps = config["fs"], config["seconds"], config["eps"]
    warmup = int(round(config["warmup_seconds"] * fs))
    rows = []
    for condition in config["test"]["conditions"]:
        change_time = condition.get("change_time")
        change_index = None if change_time is None else int(round(change_time * fs))
        # condition.get("key", default) returns the default when the key is absent.
        for seed in expand_seeds(config["test"]["seeds"]):
            clean, measured, reference = make_ecg(
                seed, seconds=seconds, fs=fs, scale=condition.get("scale", 1.0),
                pathway_after=condition.get("pathway_after"), change_time=change_time,
                leakage=condition.get("leakage", 0.0))
            for method in METHODS:
                metrics = evaluate_case(method, clean, measured, reference, params[method],
                                        warmup, fs=fs, eps=eps, change_index=change_index)
                rows.append({"condition": condition["name"], "seed": seed, "method": method,
                             **metrics})
          
                # **metrics unpacks all metric entries into this row.
    return rows

def _finite_mean(values):
    """Mean of the finite values only (nan if there are none)."""
   
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else np.nan


def summarise(rows, condition_names):
    """Per-method means and paired differences (NLMS - LMS, NLMS - ablation) per condition.

    Returns
    -------
    summary : list of dict (One row per condition x method with mean metrics and divergence count.)
    paired : list of dict (One row per condition x comparison x metric, from paired_difference.)
    """
    
    summary, paired = [], []
    for name in condition_names:
        by_method = {method: sorted((r for r in rows if r["condition"] == name and r["method"] == method),
                                    key=lambda r: r["seed"])
                     for method in METHODS}
       
        # Sort by seed so that the i-th result of each method comes from the same recording.
        for method in METHODS:
            cases = by_method[method]
            summary.append({
                "condition": name, "method": method, "n_cases": len(cases),
                "n_diverged": sum(r["diverged"] for r in cases),
                **{metric: _finite_mean([r[metric] for r in cases])
                   for metric in ("snr_in_db", "snr_out_db", "peak_error",
                                  "snr_warmup_db", "snr_post_change_db")},
            })
        for a, b in (("nlms", "lms"), ("nlms", "nlms_ablation")):
            for metric in ("snr_out_db", "peak_error"):
                stats = paired_difference([r[metric] for r in by_method[a]],
                                          [r[metric] for r in by_method[b]])
                paired.append({"condition": name, "comparison": f"{a} - {b}",
                               "metric": metric, **stats})
    return summary, paired

def describe_environment():
    """Python and package versions, operating system and processor of this machine."""
    
    machine = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            machine = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                     capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    
    # If the query fails, keep the value from platform rather than stopping the program.
    return {"python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "os": platform.platform(), "machine": machine}


def time_all_methods(config, params):
    """Time each method on the first test recordings at the nominal scale 1.0.

    Returns
    -------
    list of dict (One row per method with median and quartile times per recording.)
    """
    # Time the three methods on the first few test recordings (scale 1.0); this measures
    # speed only and selects nothing, so it causes no information leakage.
    fs, seconds = config["fs"], config["seconds"]
    seeds = expand_seeds(config["test"]["seeds"])[:config["timing"]["recordings"]]
    recordings = [make_ecg(seed, seconds=seconds, fs=fs) for seed in seeds]
    rows = []
    for method in METHODS:
        stats = time_method(method, recordings, params[method], config["eps"],
                            repeats=config["timing"]["repeats"])
        rows.append({"method": method, "M": params[method]["M"],
                     "samples_per_recording": int(seconds * fs), "setup_included": False,
                     **stats})
    return rows

def main():
    """Command-line entry point."""
    # Command-line entry point: read arguments, select parameters, save the results.
    parser = argparse.ArgumentParser(description="Compare LMS, NLMS and the ablation.")
    parser.add_argument("--config", required=True, help="path to a JSON configuration file")
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_config(args.config)
    out_dir = ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    # Create the output folder automatically; no error if it already exists.

    selected, table = select_all_parameters(config)
    (out_dir / "selected_params.json").write_text(json.dumps(selected, indent=2))
    write_csv(out_dir / "validation_grid.csv", table)
    for method, params in selected.items():
        print(f"{method:14s} {params}")

    rows = evaluate_test_cases(config, selected)
    write_csv(out_dir / "cases.csv", rows)
    print(f"evaluated {len(rows)} cases -> {out_dir / 'cases.csv'}")
    # Evaluate on the test set and save one row per case.

    names = [condition["name"] for condition in config["test"]["conditions"]]
    summary, paired = summarise(rows, names)
    write_csv(out_dir / "summary.csv", summary)
    write_csv(out_dir / "paired.csv", paired)
    print("\nNLMS - LMS, output SNR (dB): mean [95% CI]")
    for row in paired:
        if row["comparison"] == "nlms - lms" and row["metric"] == "snr_out_db":
            print(f"  {row['condition']:15s} {row['mean']:+6.2f} [{row['ci_low']:+6.2f}, {row['ci_high']:+6.2f}]")
    # Summarise, save both tables and print the main result: the NLMS - LMS difference in
    # output SNR for each condition.


    timing = time_all_methods(config, selected)
    environment = describe_environment()
    write_csv(out_dir / "timing.csv", timing)
    (out_dir / "environment.json").write_text(json.dumps(environment, indent=2))
    (out_dir / "config_used.json").write_text(json.dumps(config, indent=2))
    print("\nRuntime per recording (median):")
    for row in timing:
        print(f"  {row['method']:14s} {row['median_s'] * 1e3:6.2f} ms")
    print(f"machine: {environment['machine']}, Python {environment['python']}")
    print(f"done in {time.perf_counter() - started:.1f} s")
 
    # Time the methods, record the environment, save the configuration used, and print
    # the runtimes and the total time.

if __name__ == "__main__":
    main()
# main() runs only when this file is executed directly, not when it is imported by tests.