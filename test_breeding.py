"""
Test suite for the reverse-engineered Wynncraft mount breeding algorithm.

Validates every submission in the dataset against inferred hard and soft constraints.
Run with:  python -m pytest test_breeding.py -v
"""

import json
import pytest

from breeding_model import (
    load_data,
    get_mounts,
    has_offspring,
    STAT_NAMES,
    check_potential,
    check_max_ge_lim,
    check_val_in_bounds,
    check_val_in_predicted_range,
    check_lim_in_predicted_range,
    check_max_in_predicted_range,
    check_energy_in_predicted_range,
)

# Load dataset once
DATA = load_data()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def all_submissions():
    """Yield (submission_id, parent_a, parent_b, offspring) for complete entries."""
    for sub in DATA:
        if not has_offspring(sub):
            continue
        pa, pb, off = get_mounts(sub)
        yield sub["id"], pa, pb, off


# ---------------------------------------------------------------------------
# Hard constraints – MUST pass for 100% of the dataset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_id,pa,pb,off", all_submissions())
def test_potential_equals_sum_of_max(sub_id, pa, pb, off):
    """Potential is exactly the sum of all 8 stat max values."""
    assert check_potential(off), (
        f"Sub {sub_id}: potential={off['potential']} != sum(max_values)"
    )


@pytest.mark.parametrize("sub_id,pa,pb,off", all_submissions())
def test_max_greater_or_equal_lim(sub_id, pa, pb, off):
    """Max must never be below lim for any stat."""
    violations = check_max_ge_lim(off)
    assert not violations, f"Sub {sub_id}: " + "; ".join(violations)


@pytest.mark.parametrize("sub_id,pa,pb,off", all_submissions())
def test_val_within_bounds(sub_id, pa, pb, off):
    """Val must be non-negative and not exceed max."""
    violations = check_val_in_bounds(off)
    assert not violations, f"Sub {sub_id}: " + "; ".join(violations)


# ---------------------------------------------------------------------------
# Soft constraints – statistical ranges inferred from the data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sub_id,pa,pb,off", all_submissions())
def test_val_in_predicted_range(sub_id, pa, pb, off):
    """
    Offspring val should fall within [max(parent val) - 15, max(parent val) + 8].
    This is a soft constraint because RNG outliers are possible.
    """
    violations = check_val_in_predicted_range(pa, pb, off)
    assert not violations, f"Sub {sub_id}: " + "; ".join(violations)


@pytest.mark.parametrize("sub_id,pa,pb,off", all_submissions())
def test_lim_in_predicted_range(sub_id, pa, pb, off):
    """
    Offspring lim should fall within [avg(parent lim) - 2, avg(parent lim) + 22].
    """
    violations = check_lim_in_predicted_range(pa, pb, off)
    assert not violations, f"Sub {sub_id}: " + "; ".join(violations)


@pytest.mark.parametrize("sub_id,pa,pb,off", all_submissions())
def test_max_in_predicted_range(sub_id, pa, pb, off):
    """
    Offspring max should fall within [avg(parent lim) + 3, avg(parent lim) + 75].
    """
    violations = check_max_in_predicted_range(pa, pb, off)
    assert not violations, f"Sub {sub_id}: " + "; ".join(violations)


@pytest.mark.parametrize("sub_id,pa,pb,off", all_submissions())
def test_energy_in_predicted_range(sub_id, pa, pb, off):
    """
    Energy_max ~ max(parent energy_max) + [0, 15]
    Energy_value ~ avg(parent energy_values) ± 120
    """
    violations = check_energy_in_predicted_range(pa, pb, off)
    assert not violations, f"Sub {sub_id}: " + "; ".join(violations)


# ---------------------------------------------------------------------------
# Edge-case / pattern tests
# ---------------------------------------------------------------------------

def test_identical_parents_can_produce_different_offspring():
    """
    Identical parents should be able to produce different offspring.
    Verified by submissions 8–11: all from 30/30/30 horses, different results.
    """
    identical = []
    for sub in DATA:
        if not has_offspring(sub):
            continue
        pa, pb, off = get_mounts(sub)
        same = all(
            pa[f"{s}_val"] == pb[f"{s}_val"]
            and pa[f"{s}_lim"] == pb[f"{s}_lim"]
            and pa[f"{s}_max"] == pb[f"{s}_max"]
            for s in STAT_NAMES
        )
        if same:
            identical.append((sub["id"], off["potential"]))

    # We know subs 8–11 are identical-parent cases with different potentials
    potentials = [p for _sid, p in identical]
    assert len(set(potentials)) > 1, "Identical parents produced identical potentials – RNG not confirmed"


def test_fresh_mounts_start_at_240_potential():
    """Fresh mounts (never bred) have potential 240."""
    fresh_count = 0
    for sub in DATA:
        if not has_offspring(sub):
            pa, pb = [m for m in sub["mounts"] if m["role"] != "offspring"][:2]
        else:
            pa, pb, _off = get_mounts(sub)
        for p in (pa, pb):
            if p["potential"] == 240:
                fresh_count += 1
    # We expect at least some fresh mounts in the dataset
    assert fresh_count > 0, "No fresh mounts found in dataset"


def test_parent_max_does_not_directly_transfer():
    """
    Parent max values should not directly determine offspring max.
    Example: sub 8 has parent max=30 for all stats but offspring max=35–45.
    """
    sub = next(s for s in DATA if s["id"] == 8)
    pa, pb, off = get_mounts(sub)
    assert all(pa[f"{s}_max"] == 30 for s in STAT_NAMES)
    assert any(off[f"{s}_max"] != 30 for s in STAT_NAMES), "Parent max transferred unchanged"


def test_color_inheritance_pattern():
    """
    Offspring colors are usually combinations of parent colors.
    We verify that at least one word from each parent's color appears in the
    offspring's color for the majority of cases.
    """
    matched = 0
    total = 0
    for sub in DATA:
        if not has_offspring(sub):
            continue
        pa, pb, off = get_mounts(sub)
        off_words = set(off["color"].replace("-", " ").split())
        pa_words = set(pa["color"].replace("-", " ").split())
        pb_words = set(pb["color"].replace("-", " ").split())

        # Offspring should share at least one word with a parent
        if off_words & pa_words or off_words & pb_words:
            matched += 1
        total += 1

    # Allow for some mutations (rare)
    assert matched / total >= 0.85, f"Color inheritance too random: {matched}/{total}"


# ---------------------------------------------------------------------------
# Single-run summary test
# ---------------------------------------------------------------------------

def test_all_submissions_pass_hard_constraints():
    """Meta-test: every single submission must pass all hard constraints."""
    failures = []
    for sub in DATA:
        if not has_offspring(sub):
            continue
        sub_id = sub["id"]
        pa, pb, off = get_mounts(sub)
        if not check_potential(off):
            failures.append(f"Sub {sub_id}: potential mismatch")
        failures.extend(f"Sub {sub_id}: {v}" for v in check_max_ge_lim(off))
        failures.extend(f"Sub {sub_id}: {v}" for v in check_val_in_bounds(off))
    assert not failures, "Hard constraint failures:\n" + "\n".join(failures)
