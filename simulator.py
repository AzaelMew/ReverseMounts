#!/usr/bin/env python3
"""
Mount Breeding Simulator – Compare reverse-engineered predictions vs real outcomes.

Modes:
    compare   Run N simulations for every submission in the dataset and report
              how often the real offspring falls inside the predicted ranges.
    predict   Accept two parent stat blocks from stdin/CLI and print predicted
              offspring ranges.

Usage:
    python simulator.py compare --runs 1000
    python simulator.py predict --parent-a '{"speed_val":30,...}' --parent-b '{"speed_val":30,...}'
"""

import argparse
import json
import math
import random
import statistics
import sys
from typing import List, Tuple, Dict

from breeding_model import (
    load_data,
    get_mounts,
    has_offspring,
    STAT_NAMES,
)

random.seed(42)

# ---------------------------------------------------------------------------
# Simulation distributions (derived from observed data)
# ---------------------------------------------------------------------------

def sim_val_bonus() -> int:
    """Observed: mean=0.64, stdev=2.12, min=-13, max=6."""
    while True:
        v = int(random.gauss(mu=0.6, sigma=2.1))
        if -13 <= v <= 6:
            return v


def sim_lim_bonus() -> int:
    """Observed: mean=9.67, stdev=3.39, min=-1, max=20."""
    while True:
        v = int(random.gauss(mu=9.7, sigma=3.4))
        if -1 <= v <= 20:
            return v


def sim_max_bonus() -> int:
    """Observed: mean=29.02, stdev=11.00, min=5, max=69."""
    while True:
        v = int(random.gauss(mu=29.0, sigma=11.0))
        if 5 <= v <= 69:
            return v


def sim_energy_max_bonus() -> int:
    """Observed: mean=1.03, stdev=2.57, min=0, max=13."""
    while True:
        v = int(random.gauss(mu=1.0, sigma=2.6))
        if 0 <= v <= 13:
            return v


def sim_energy_value_diff() -> int:
    """Observed: min=-113, max=13, but most are close to average.
    We model this as a wide normal with a bias toward 0."""
    while True:
        v = int(random.gauss(mu=0.0, sigma=25.0))
        if -120 <= v <= 20:
            return v


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_offspring(parent_a: dict, parent_b: dict) -> dict:
    """Generate a single simulated offspring from two parents."""
    offspring = {}

    for stat in STAT_NAMES:
        # val
        base_val = max(parent_a[f"{stat}_val"], parent_b[f"{stat}_val"])
        offspring[f"{stat}_val"] = base_val + sim_val_bonus()

        # lim
        avg_lim = (parent_a[f"{stat}_lim"] + parent_b[f"{stat}_lim"]) / 2.0
        offspring[f"{stat}_lim"] = int(avg_lim) + sim_lim_bonus()

        # max (must be >= lim)
        raw_max = int(avg_lim) + sim_max_bonus()
        offspring[f"{stat}_max"] = max(raw_max, offspring[f"{stat}_lim"])

        # Clamp val so it stays within [0, max]
        offspring[f"{stat}_val"] = max(0, min(offspring[f"{stat}_val"], offspring[f"{stat}_max"]))

    # Energy
    offspring["energy_max"] = max(parent_a["energy_max"], parent_b["energy_max"]) + sim_energy_max_bonus()
    avg_ev = (parent_a["energy_value"] + parent_b["energy_value"]) / 2.0
    offspring["energy_value"] = int(avg_ev) + sim_energy_value_diff()
    offspring["energy_value"] = max(0, min(offspring["energy_value"], offspring["energy_max"]))

    # Potential
    offspring["potential"] = sum(offspring[f"{s}_max"] for s in STAT_NAMES)

    return offspring


def simulate_many(parent_a: dict, parent_b: dict, runs: int = 1000) -> Dict[str, List[int]]:
    """Run N simulations and collect per-stat distributions."""
    results: Dict[str, List[int]] = {f"{s}_{k}": [] for s in STAT_NAMES for k in ("val", "lim", "max")}
    results["potential"] = []
    results["energy_max"] = []
    results["energy_value"] = []

    for _ in range(runs):
        off = simulate_offspring(parent_a, parent_b)
        for s in STAT_NAMES:
            results[f"{s}_val"].append(off[f"{s}_val"])
            results[f"{s}_lim"].append(off[f"{s}_lim"])
            results[f"{s}_max"].append(off[f"{s}_max"])
        results["potential"].append(off["potential"])
        results["energy_max"].append(off["energy_max"])
        results["energy_value"].append(off["energy_value"])

    return results


# ---------------------------------------------------------------------------
# Comparison mode
# ---------------------------------------------------------------------------

def summarize_distribution(values: List[int]) -> Tuple[int, int, int, float]:
    """Return (min, max, mean, stdev) for a list of ints."""
    return min(values), max(values), int(round(statistics.mean(values))), statistics.stdev(values)


def compare_mode(runs: int = 1000):
    data = load_data()
    complete = [s for s in data if has_offspring(s)]

    hits = 0
    total = 0
    potential_hits = 0

    print(f"Running {runs} simulations per submission ({len(complete)} submissions)...\n")

    for sub in complete:
        sub_id = sub["id"]
        pa, pb, off = get_mounts(sub)
        sims = simulate_many(pa, pb, runs)

        print(f"--- Submission {sub_id} ---")
        print(f"{'Stat':<12} {'Actual':>8} {'PredMin':>8} {'PredMax':>8} {'PredMean':>8} {'PredSD':>8} {'Hit'}")

        # Potential
        pmin, pmax, pmean, psd = summarize_distribution(sims["potential"])
        phit = pmin <= off["potential"] <= pmax
        potential_hits += int(phit)
        print(f"{'potential':<12} {off['potential']:>8} {pmin:>8} {pmax:>8} {pmean:>8} {psd:>8.1f} {'Y' if phit else 'N'}")

        for stat in STAT_NAMES:
            for field in ("val", "lim", "max"):
                key = f"{stat}_{field}"
                actual = off[key]
                vmin, vmax, vmean, vsd = summarize_distribution(sims[key])
                hit = vmin <= actual <= vmax
                hits += int(hit)
                total += 1
                print(f"{key:<12} {actual:>8} {vmin:>8} {vmax:>8} {vmean:>8} {vsd:>8.1f} {'Y' if hit else 'N'}")

        # Energy
        for key in ("energy_max", "energy_value"):
            actual = off[key]
            vmin, vmax, vmean, vsd = summarize_distribution(sims[key])
            hit = vmin <= actual <= vmax
            hits += int(hit)
            total += 1
            print(f"{key:<12} {actual:>8} {vmin:>8} {vmax:>8} {vmean:>8} {vsd:>8.1f} {'Y' if hit else 'N'}")

        print()

    print("=" * 60)
    print(f"Overall stat hit rate : {hits}/{total} = {hits/total*100:.1f}%")
    print(f"Potential hit rate    : {potential_hits}/{len(complete)} = {potential_hits/len(complete)*100:.1f}%")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Prediction mode
# ---------------------------------------------------------------------------

def predict_mode(parent_a: dict, parent_b: dict, runs: int = 10000):
    sims = simulate_many(parent_a, parent_b, runs)

    print(f"\nPredicted offspring after {runs} simulations:")
    print(f"{'Stat':<12} {'Min':>6} {'Max':>6} {'Mean':>6} {'SD':>6}")
    print("-" * 40)

    pmin, pmax, pmean, psd = summarize_distribution(sims["potential"])
    print(f"{'potential':<12} {pmin:>6} {pmax:>6} {pmean:>6} {psd:>6.1f}")

    for stat in STAT_NAMES:
        for field in ("val", "lim", "max"):
            key = f"{stat}_{field}"
            vmin, vmax, vmean, vsd = summarize_distribution(sims[key])
            print(f"{key:<12} {vmin:>6} {vmax:>6} {vmean:>6} {vsd:>6.1f}")

    for key in ("energy_max", "energy_value"):
        vmin, vmax, vmean, vsd = summarize_distribution(sims[key])
        print(f"{key:<12} {vmin:>6} {vmax:>6} {vmean:>6} {vsd:>6.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_parent(raw: str) -> dict:
    """Parse a JSON string or a shorthand 'val/lim/max' block into a parent dict."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise SystemExit(f"Invalid JSON for parent: {raw}")


def main():
    parser = argparse.ArgumentParser(description="Wynncraft Mount Breeding Simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # compare
    p_cmp = subparsers.add_parser("compare", help="Compare predictions against dataset")
    p_cmp.add_argument("--runs", type=int, default=1000, help="Simulations per submission")

    # predict
    p_pred = subparsers.add_parser("predict", help="Predict offspring from two parents")
    p_pred.add_argument("--parent-a", required=True, help="JSON dict of parent A stats")
    p_pred.add_argument("--parent-b", required=True, help="JSON dict of parent B stats")
    p_pred.add_argument("--runs", type=int, default=10000, help="Simulations to run")

    args = parser.parse_args()

    if args.command == "compare":
        compare_mode(args.runs)
    elif args.command == "predict":
        pa = parse_parent(args.parent_a)
        pb = parse_parent(args.parent_b)
        predict_mode(pa, pb, args.runs)


if __name__ == "__main__":
    main()
