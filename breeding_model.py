"""
Reverse-engineered Wynncraft mount breeding algorithm.

Based on analysis of 35 breeding submissions. The game uses RNG, so
offspring stats fall within statistical ranges rather than being deterministic.

Key findings:
- potential = sum of all 8 stat max values (hard rule, no exceptions)
- val  ~ max(parent vals)  + random variance (usually 0 to +6)
- lim  ~ avg(parent lims)  + random bonus (usually +5 to +15)
- max  ~ avg(parent lims)  + random bonus (usually +15 to +45)
- energy_max ~ max(parent energy_max) + small bonus (usually 0 to +5)
- energy_value ~ avg(parent energy_values) with variance
- parent max values do NOT affect offspring
"""

import json
from dataclasses import dataclass
from typing import List, Tuple

STAT_NAMES = [
    "speed", "accel", "altitude", "energy_stat",
    "handling", "toughness", "boost", "training",
]


def load_data(path: str = "wynnbreeder_export (1).json") -> List[dict]:
    with open(path) as f:
        return json.load(f)


def get_mounts(submission: dict) -> Tuple[dict, dict, dict]:
    """Return (parent_a, parent_b, offspring) from a submission.
    Raises KeyError if offspring is missing (e.g., pending breed)."""
    mounts = {m["role"]: m for m in submission["mounts"]}
    return mounts["parent_a"], mounts["parent_b"], mounts["offspring"]


def has_offspring(submission: dict) -> bool:
    """True if the submission includes an offspring mount."""
    return any(m["role"] == "offspring" for m in submission["mounts"])


# ---------------------------------------------------------------------------
# Prediction helpers – return (low, high) inclusive bounds
# ---------------------------------------------------------------------------

def predict_val_range(parent_a: dict, parent_b: dict, stat: str) -> Tuple[int, int]:
    """Expected offspring val range for a stat."""
    mx = max(parent_a[f"{stat}_val"], parent_b[f"{stat}_val"])
    # Observed across 280 stat-samples: min -13, max +6, 5th pct 0, 95th pct 4
    # We use generous bounds to account for edge cases.
    return (mx - 15, mx + 8)


def predict_lim_range(parent_a: dict, parent_b: dict, stat: str) -> Tuple[int, int]:
    """Expected offspring lim range for a stat."""
    avg = (parent_a[f"{stat}_lim"] + parent_b[f"{stat}_lim"]) / 2
    # Observed: min -1, max +20, mean +9.7, stdev 3.4
    return (int(avg) - 2, int(avg) + 22)


def predict_max_range(parent_a: dict, parent_b: dict, stat: str) -> Tuple[int, int]:
    """Expected offspring max range for a stat."""
    avg = (parent_a[f"{stat}_lim"] + parent_b[f"{stat}_lim"]) / 2
    # Observed: min +5, max +69, mean +29, stdev 11
    return (int(avg) + 3, int(avg) + 75)


def predict_energy_max_range(parent_a: dict, parent_b: dict) -> Tuple[int, int]:
    """Expected offspring energy_max range."""
    mx = max(parent_a["energy_max"], parent_b["energy_max"])
    # Observed: 0 to +13, mean +1, stdev 2.6
    return (mx, mx + 15)


def predict_energy_value_range(parent_a: dict, parent_b: dict) -> Tuple[int, int]:
    """Expected offspring energy_value range."""
    avg = (parent_a["energy_value"] + parent_b["energy_value"]) / 2
    # Very wide because one parent can have near-zero energy.
    return (int(avg) - 120, int(avg) + 20)


# ---------------------------------------------------------------------------
# Hard constraints (must be true for every submission)
# ---------------------------------------------------------------------------

def check_potential(offspring: dict) -> bool:
    """potential == sum of all 8 stat max values."""
    total = sum(offspring[f"{s}_max"] for s in STAT_NAMES)
    return total == offspring["potential"]


def check_max_ge_lim(offspring: dict) -> List[str]:
    """max must be >= lim for every stat."""
    violations = []
    for s in STAT_NAMES:
        if offspring[f"{s}_max"] < offspring[f"{s}_lim"]:
            violations.append(f"{s}: max={offspring[f'{s}_max']} < lim={offspring[f'{s}_lim']}")
    return violations


def check_val_in_bounds(offspring: dict) -> List[str]:
    """0 <= val <= max for every stat."""
    violations = []
    for s in STAT_NAMES:
        v = offspring[f"{s}_val"]
        if v < 0 or v > offspring[f"{s}_max"]:
            violations.append(f"{s}: val={v} not in [0, {offspring[f'{s}_max']}]")
    return violations


# ---------------------------------------------------------------------------
# Soft constraints (statistical ranges derived from data)
# ---------------------------------------------------------------------------

def check_val_in_predicted_range(parent_a: dict, parent_b: dict, offspring: dict) -> List[str]:
    """Check offspring val falls within predicted range."""
    violations = []
    for s in STAT_NAMES:
        low, high = predict_val_range(parent_a, parent_b, s)
        v = offspring[f"{s}_val"]
        if v < low or v > high:
            violations.append(
                f"{s}: val={v} not in predicted range [{low}, {high}] "
                f"(parents {parent_a[f'{s}_val']}/{parent_b[f'{s}_val']})"
            )
    return violations


def check_lim_in_predicted_range(parent_a: dict, parent_b: dict, offspring: dict) -> List[str]:
    """Check offspring lim falls within predicted range."""
    violations = []
    for s in STAT_NAMES:
        low, high = predict_lim_range(parent_a, parent_b, s)
        v = offspring[f"{s}_lim"]
        if v < low or v > high:
            violations.append(
                f"{s}: lim={v} not in predicted range [{low}, {high}] "
                f"(parents {parent_a[f'{s}_lim']}/{parent_b[f'{s}_lim']})"
            )
    return violations


def check_max_in_predicted_range(parent_a: dict, parent_b: dict, offspring: dict) -> List[str]:
    """Check offspring max falls within predicted range."""
    violations = []
    for s in STAT_NAMES:
        low, high = predict_max_range(parent_a, parent_b, s)
        v = offspring[f"{s}_max"]
        if v < low or v > high:
            violations.append(
                f"{s}: max={v} not in predicted range [{low}, {high}] "
                f"(parents {parent_a[f'{s}_lim']}/{parent_b[f'{s}_lim']})"
            )
    return violations


def check_energy_in_predicted_range(parent_a: dict, parent_b: dict, offspring: dict) -> List[str]:
    """Check offspring energy fields fall within predicted range."""
    violations = []
    low, high = predict_energy_max_range(parent_a, parent_b)
    v = offspring["energy_max"]
    if v < low or v > high:
        violations.append(f"energy_max={v} not in predicted range [{low}, {high}]")

    low, high = predict_energy_value_range(parent_a, parent_b)
    v = offspring["energy_value"]
    if v < low or v > high:
        violations.append(f"energy_value={v} not in predicted range [{low}, {high}]")
    return violations


# ---------------------------------------------------------------------------
# Summary statistics generator
# ---------------------------------------------------------------------------

def compute_bonus_stats(data: List[dict]):
    """Print observed bonus distributions (useful for model tuning)."""
    val_b, lim_b, max_b, em_b = [], [], [], []
    for sub in data:
        pa, pb, off = get_mounts(sub)
        for s in STAT_NAMES:
            mx = max(pa[f"{s}_val"], pb[f"{s}_val"])
            val_b.append(off[f"{s}_val"] - mx)

            avg = (pa[f"{s}_lim"] + pb[f"{s}_lim"]) / 2
            lim_b.append(off[f"{s}_lim"] - avg)
            max_b.append(off[f"{s}_max"] - avg)

        em_b.append(off["energy_max"] - max(pa["energy_max"], pb["energy_max"]))

    for name, vals in [("val_bonus", val_b), ("lim_bonus", lim_b),
                       ("max_bonus", max_b), ("energy_max_bonus", em_b)]:
        print(f"{name}: min={min(vals):.1f} max={max(vals):.1f} mean={sum(vals)/len(vals):.2f}")


if __name__ == "__main__":
    data = load_data()
    compute_bonus_stats(data)
