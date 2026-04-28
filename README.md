# ReverseMounts – Wynncraft Mount Breeding Algorithm

This repository contains a reverse-engineered statistical model of the **Wynncraft** mount (horse/wyvern) breeding system. It includes the raw breeding data, a validation test suite, and a Monte Carlo simulator that can predict offspring outcomes from parent stats.

---

## Table of Contents

- [Background](#background)
- [Dataset](#dataset)
- [Reverse-Engineered Algorithm](#reverse-engineered-algorithm)
- [Repository Files](#repository-files)
- [Running the Test Suite](#running-the-test-suite)
- [Using the Simulator](#using-the-simulator)
  - [Compare Mode](#compare-mode)
  - [Predict Mode](#predict-mode)
- [Key Findings](#key-findings)
- [Model Limitations](#model-limitations)

---

## Background

In Wynncraft, mounts (horses, wyverns, adasaurs) have **8 stats**. Each stat has three values:

| Field | Meaning |
|-------|---------|
| `val` | Current trained level |
| `lim` | Feeding cap (you can't feed past this) |
| `max` | Absolute ceiling (can only be raised by breeding again) |

When you breed two mounts, both parents are consumed and you get **one offspring**. The game uses RNG, so the exact outcome is probabilistic—but the general formulas are discoverable from enough data.

This repo analyzes **35 real breeding submissions** (horses and wyverns) and extracts the underlying mechanics.

---

## Dataset

- **File**: `wynnbreeder_export (1).json`
- **Submissions**: 39 total (32 complete with offspring, 7 pending)
- **Mount types**: Horse, Wyvern

Each submission is a JSON object with three mounts:

```json
{
  "id": 1,
  "status": "complete",
  "mounts": [
    { "role": "offspring", "type": "horse", "potential": 425, ... },
    { "role": "parent_a",  "type": "horse", "potential": 362, ... },
    { "role": "parent_b",  "type": "horse", "potential": 359, ... }
  ]
}
```

Each mount has these fields:

```
potential, color, name,
energy_value, energy_max,
speed_val, speed_lim, speed_max,
accel_val, accel_lim, accel_max,
altitude_val, altitude_lim, altitude_max,
energy_stat_val, energy_stat_lim, energy_stat_max,
handling_val, handling_lim, handling_max,
toughness_val, toughness_lim, toughness_max,
boost_val, boost_lim, boost_max,
training_val, training_lim, training_max
```

---

## Reverse-Engineered Algorithm

After analyzing all submissions, these are the inferred formulas:

### 1. Potential (Hard Rule)
```
potential = Σ stat_max   (sum of all 8 stat max values)
```
**Verified 100%** across the entire dataset. No exceptions.

### 2. Offspring `val` (Current Level)
```
val = max(parent_a.val, parent_b.val) + random_variance
```
- **Typical variance**: `+0` to `+6`
- **Extreme range**: `-13` to `+6`
- When parents are identical, offspring `val` can still exceed parent `val` by 1–6 points.
- Negative variance happens when parents have very different levels or very high stats.

### 3. Offspring `lim` (Level Limit)
```
lim = avg(parent_a.lim, parent_b.lim) + random_bonus
```
- **Typical bonus**: `+5` to `+15`
- **Extreme range**: `-1` to `+20`
- **Parent max values do NOT affect offspring limit.** Only parent limits matter.

### 4. Offspring `max` (Absolute Ceiling)
```
max = avg(parent_a.lim, parent_b.lim) + random_bonus
max = max(max, lim)   // max can never be below lim
```
- **Typical bonus**: `+15` to `+45`
- **Extreme range**: `+5` to `+69`
- Again, **parent maxes don't transfer**—only parent limits drive the offspring max.
- Rarely, `max == lim` for all stats (observed in 3 low-potential breeds from identical fresh parents).

### 5. Energy
```
energy_max   = max(parent_a.energy_max, parent_b.energy_max) + [0, +5]
energy_value ≈ avg(parent_a.energy_value, parent_b.energy_value) ± variance
```

### 6. Color
Offspring colors are usually a combination of parent colors (e.g. `Bay-Dawn` × `Gray-Reddish` → `Bay-Reddish`). There is a small chance for mutation producing a color not present on either parent.

### 7. Randomness
The exact mechanics involve **per-stat RNG**. Identical parents can produce wildly different offspring. Verified: four breeds from identical 30/30/30 horses produced potentials of **309, 329, 333, and 460**.

---

## Repository Files

| File | Description |
|------|-------------|
| `wynnbreeder_export (1).json` | Raw breeding data (35 complete submissions) |
| `breeding_model.py` | Statistical model with prediction ranges and constraint checkers |
| `test_breeding.py` | `pytest` suite (229 tests) validating every submission |
| `simulator.py` | Monte Carlo simulator with `compare` and `predict` modes |
| `README.md` | This file |
| `.gitignore` | Excludes `venv/` and `__pycache__/` |

---

## Running the Test Suite

```bash
# Create virtual environment (first time only)
python -m venv venv
source venv/bin/activate

# Install pytest
pip install pytest

# Run all tests
python -m pytest test_breeding.py -v
```

### Test Categories

- **Hard constraints** (must pass for 100% of data):
  - `potential == sum(max_values)`
  - `max >= lim` for all stats
  - `0 <= val <= max` for all stats
- **Soft constraints** (statistical ranges, derived from observed data):
  - `val` within predicted range
  - `lim` within predicted range
  - `max` within predicted range
  - `energy` fields within predicted range
- **Edge-case tests**:
  - Identical parents produce different offspring
  - Fresh mounts have potential 240
  - Parent max does not directly transfer
  - Color inheritance heuristic

Current status: **229/229 tests passing**.

---

## Using the Simulator

The simulator (`simulator.py`) uses Gaussian distributions fitted to the observed data to generate offspring. It supports two modes:

### Compare Mode

Validates the model against the real dataset. For each submission, it runs N simulations and reports whether the actual offspring falls inside the predicted min/max range.

```bash
python simulator.py compare --runs 1000
```

Example output summary:
```
============================================================
Overall stat hit rate : 825/832 = 99.2%
Potential hit rate    : 29/32 = 90.6%
============================================================
```

The misses are mostly the rare `lim=max` edge cases (submissions 8–10) where RNG produced abnormally low potential, plus a couple of energy outliers.

### Predict Mode

Takes two parent stat blocks (as JSON) and runs thousands of Monte Carlo simulations to show predicted offspring ranges.

```bash
python simulator.py predict \
  --parent-a '{"speed_val":30,"speed_lim":30,"speed_max":30,"accel_val":30,"accel_lim":30,"accel_max":30,"altitude_val":30,"altitude_lim":30,"altitude_max":30,"energy_stat_val":30,"energy_stat_lim":30,"energy_stat_max":30,"handling_val":30,"handling_lim":30,"handling_max":30,"toughness_val":30,"toughness_lim":30,"toughness_max":30,"boost_val":30,"boost_lim":30,"boost_max":30,"training_val":30,"training_lim":30,"training_max":30,"energy_value":191,"energy_max":259}' \
  --parent-b '{"speed_val":30,"speed_lim":30,"speed_max":30,"accel_val":30,"accel_lim":30,"accel_max":30,"altitude_val":30,"altitude_lim":30,"altitude_max":30,"energy_stat_val":30,"energy_stat_lim":30,"energy_stat_max":30,"handling_val":30,"handling_lim":30,"handling_max":30,"toughness_val":30,"toughness_lim":30,"toughness_max":30,"boost_val":30,"boost_lim":30,"boost_max":30,"training_val":30,"training_lim":30,"training_max":30,"energy_value":191,"energy_max":259}' \
  --runs 10000
```

Example output:
```
Predicted offspring after 10000 simulations:
Stat            Min    Max   Mean     SD
----------------------------------------
potential       370    583    472   29.0
speed_val        24     36     30    1.7
speed_lim        29     50     39    3.3
speed_max        35     98     59   10.4
...
energy_max      259    269    261    1.8
energy_value     97    211    183   18.7
```

This tells you that breeding two identical fresh 30/30/30 horses will most likely produce an offspring with:
- **Potential**: 370–583 (mean ~472)
- **Per-stat lim**: ~39 (range 29–50)
- **Per-stat max**: ~59 (range 35–98)

---

## Key Findings

1. **Potential is deterministic**: `sum of all 8 max values`. Always.
2. **Parent maxes don't matter**: Only parent limits drive offspring limits and maxes.
3. **Feed both parents equally**: The average of parent limits is what counts. A high-limit parent bred with a low-limit parent will drag the average down.
4. **Train one parent, not both**: Offspring `val` is based on the higher parent's `val`, so you only need to train one mount.
5. **RNG is significant**: Even with perfect identical parents, you can get a 309-potential dud or a 460-potential stud.
6. **lim=max edge case exists**: Rarely, offspring max equals lim for all stats, creating a "dead-end" breed that can't grow further through feeding.

---

## Model Limitations

- The exact RNG distributions are **approximations** based on 35 submissions (~280 stat observations). More data would tighten the ranges.
- Color mutation mechanics are not fully modeled—only a heuristic inheritance check is implemented.
- The simulator does not account for mount type differences (horse vs wyvern vs adasaur) because the dataset only contains horses and wyverns, and they appear to follow the same formulas.
- Adasaur mounts are not represented in the dataset at all.
