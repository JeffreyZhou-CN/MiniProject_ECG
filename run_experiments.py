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
from pathlib import Path

from src.data import make_ecg
from src.experiment import (METHODS, check_disjoint, evaluate_case, expand_seeds, mean_input_energy, select_parameters)

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


def main():
    """Command-line entry point."""
    # Command-line entry point: read arguments, select parameters, save the results.
    parser = argparse.ArgumentParser(description="Compare LMS, NLMS and the ablation.")
    parser.add_argument("--config", required=True, help="path to a JSON configuration file")
    args = parser.parse_args()

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

if __name__ == "__main__":
    main()
# main() runs only when this file is executed directly, not when it is imported by tests.