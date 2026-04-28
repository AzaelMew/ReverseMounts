# Wynncraft Mount Breeding Algorithm – Reverse Engineering

This repository contains a reverse-engineered statistical model of the Wynncraft mount (horse/wyvern) breeding algorithm, plus a test suite that validates the model against real breeding data.

## Dataset

- **File**: `wynnbreeder_export (1).json`
- **Submissions**: 35 complete breeding outcomes
- **Mount types**: Horse and Wyvern
- **Stats per mount**: 8 core stats (`speed`, `accel`, `altitude`, `energy_stat`, `handling`, `toughness`, `boost`, `training`), each with `val` (current), `lim` (limit), `max` (ceiling), plus `energy_value` and `energy_max`.

## Key Findings

### 1. Potential Formula (Hard Rule)
```
potential = Σ stat_max   (sum of all 8 stat max values)
```
Verified across 100% of the dataset without exception.

### 2. Offspring `val` (Current Level)
- Based on the **higher** parent's `val` with random variance.
- Typical range: `max(parent val) + [0, +6]`
- Edge cases: can drop by up to ~13 when parents have wildly different levels or very high stats.

### 3. Offspring `lim` (Level Limit)
- Based on the **average** of both parents' limits, plus a bonus.
- Typical range: `avg(parent lims) + [+5, +15]`
- **Parent max values do NOT affect offspring limit.**

### 4. Offspring `max` (Absolute Ceiling)
- Also derived from the **average of parents' limits** (not their maxes).
- Typical range: `avg(parent lims) + [+15, +45]`
- Always `max >= lim`.
- Rarely, `max == lim` for every stat (observed in 3 low-potential breeds from identical fresh parents).

### 5. Energy
- `energy_max` ≈ `max(parent energy_max) + [0, +5]`
- `energy_value` ≈ `avg(parent energy_values) ± variance`

### 6. Color
- Offspring colors are usually a combination of parent colors.
- Small chance of mutation producing a color not seen on either parent.

### 7. Randomness
- The exact mechanics involve **per-stat RNG**.
- Identical parents can produce very different offspring (verified: 4 breeds from identical 30/30/30 horses produced potentials 309, 329, 333, and 460).

## Files

| File | Description |
|------|-------------|
| `breeding_model.py` | Statistical model with prediction ranges and constraint checkers |
| `test_breeding.py` | `pytest` suite that validates every submission in the dataset |
| `wynnbreeder_export (1).json` | Raw breeding outcome data |

## Running the Tests

```bash
# Install pytest if needed
pip install pytest

# Run the full suite
python -m pytest test_breeding.py -v
```

### Test Categories

- **Hard constraints** – must pass for 100% of data:
  - `potential == sum(max_values)`
  - `max >= lim` for all stats
  - `0 <= val <= max` for all stats
- **Soft constraints** – statistical ranges inferred from the data:
  - `val` within predicted range
  - `lim` within predicted range
  - `max` within predicted range
  - `energy` fields within predicted range
- **Edge-case tests** – verify RNG behavior, color inheritance, fresh mounts, etc.

## Model Tuning

If you obtain more breeding data and want to refine the ranges:

```bash
python breeding_model.py
```

This prints the observed bonus distributions (`val_bonus`, `lim_bonus`, `max_bonus`, `energy_max_bonus`) which can be used to update the constants in `breeding_model.py`.
