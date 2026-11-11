# R41-T019: Test an Input Perturbation: Combined-Cycle Power Plant – Linear Regression

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic deterministic formula fixture (`fixture.json`). These records verify input perturbation robustness, row-order permutation invariance via stable IDs, and unsupported input rejection—they are **not** observations from the real Combined-cycle power plant dataset and are **not** evidence of real-world or clinical performance.

## Data Contract & Synthetic Fixture

- **Schema:** `(record_id, x1, x2, target)`
- **Deterministic Formula:** $\text{target} = 2 \cdot x1 + 1$ with $x1 = [0, 1, 2, 3, 4, 5]$ and $x2 = 1.0$.
- **Split Design:**
  - **Train:** `["r1", "r2", "r3", "r4"]`
  - **Validation:** `["r5"]`
  - **Test:** `["r6"]`

## Robustness Property Tested

1. **Row-Order Permutation Invariance:**
   - Batch input records may arrive in perturbed order.
   - When aligned by stable `record_id`, outputs are strictly preserved ($\Delta < 10^{-6}$).
2. **Rejection of Unsupported Inputs:**
   - Missing features (`x1`, `x2`) or missing `record_id` raise an explicit `ValueError`.
   - Non-numeric inputs raise an explicit `TypeError`.
3. **Scope Clarification:**
   - This test confirms structural pipeline invariance to record ordering and input validation; it does **not** claim broad adversarial robustness.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t019 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Bounded offline verification component for interface robustness.
- Any optional real-data extension must separately record source, license, access requirements, and split design.
