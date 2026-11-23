# R41-T009: Input Perturbation Robustness (Kaggle House Prices – Feature Engineering)

## Objective
Verify robustness properties of a tabular regression interface under valid and invalid input perturbations:
1. **Permutation Invariance under ID Alignment:** Changing the presentation order of evaluation records preserves exact predictions once aligned by stable `record_id`.
2. **Unsupported Input Handling:** Rejects incompatible input shapes (missing required features, boolean values, non-numeric categories) with `IncompatibleInputError`, or maps to an explicit fallback default (`4.0`).
3. **Explicit Scope Limitation:** Confirms tabular record-order invariance with stable IDs; does **not** claim broad adversarial robustness, distribution-shift resilience, or adversarial noise defense.

## Synthetic Fixture Specification
> [!IMPORTANT]
> **DISCLAIMER:** This dataset is an explicitly synthetic, deterministic formula fixture (`target = 2 * x1 + 1`) created solely for offline pipeline verification and robustness testing. It does **NOT** contain real-world Ames/Kaggle housing sales records, real-estate transactions, or actual home valuation estimates.

### Schema
| Field | Type | Description |
|---|---|---|
| `record_id` | String | Unique record ID (`r1`..`r6`) |
| `x1` | Numeric | Scalar feature `[0, 1, 2, 3, 4, 5]` |
| `x2` | Numeric | Constant feature `[1, 1, 1, 1, 1, 1]` |
| `target` | Numeric | Deterministic target `2 * x1 + 1` |

### Split Manifest
- **Training (4 rows):** `["r1", "r2", "r3", "r4"]`
- **Validation (1 row):** `["r5"]`
- **Test (1 row):** `["r6"]`

## Test Verification
Run the offline unit tests:
```bash
python3 -m unittest discover -s course-labs/r41-t009 -p "test_*.py" -v
```
All 8 tests execute offline in <0.01 seconds using Python standard library only.
