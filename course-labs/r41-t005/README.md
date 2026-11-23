# R41-T005: Explicit Preprocessing Contract (Student Performance UCI – Regression)

## Objective
Implement an explicit, leakage-free preprocessing pipeline component for regression features under a strictly verified data contract:
1. Missing-value imputation and feature scaling parameters are fitted **strictly on training rows**.
2. Fitting on held-out inputs (validation or test rows) is impossible through the public `transform()` interface.
3. Feature order (`['x1', 'x2']`) and numeric float types are strictly preserved.
4. Deliberate contamination fixtures (held-out IDs in training, corrupted schemas, non-numeric types) are rejected.

## Synthetic Fixture Specification
> [!IMPORTANT]
> **DISCLAIMER:** This dataset is an explicitly synthetic, deterministic formula fixture (`target = 2 * x1 + 1`) created solely for offline pipeline verification and preprocessing contract testing. It does **NOT** contain real-world student performance observations, UCI repository data, or real academic records.

### Schema
| Field | Type | Description |
|---|---|---|
| `record_id` | String | Unique, non-empty identifier (`r1`..`r6`) |
| `x1` | Numeric | Scalar feature `[0, 1, 2, 3, 4, 5]` |
| `x2` | Numeric | Constant feature `[1, 1, 1, 1, 1, 1]` |
| `target` | Numeric | Deterministic formula target `2 * x1 + 1` |

### Split Manifest
- **Training:** `["r1", "r2", "r3", "r4"]`
- **Validation:** `["r5"]`
- **Test:** `["r6"]`

## Preprocessing Contract & Fitted Parameters
Parameters computed strictly on training rows (`r1`..`r4`):
- **Feature `x1` (values `[0, 1, 2, 3]`):**
  - Mean: $1.5$
  - Population Variance: $1.25$
  - Population Standard Deviation: $\sqrt{1.25} \approx 1.118033988749895$
- **Feature `x2` (values `[1, 1, 1, 1]`):**
  - Mean: $1.0$
  - Population Standard Deviation: $0.0$ (centered to $0.0$)

## Test Verification
Run the offline, bounded test suite:
```bash
python3 -m unittest discover -s course-labs/r41-t005 -p "test_*.py" -v
```
All 15 tests execute offline in <0.01 seconds without external dependencies.
