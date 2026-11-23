# R41-T006: Bounded CPU Smoke Run (Forest Fires – Regression)

## Objective
Build and verify a tiny regression pipeline smoke test:
1. Enforces strict execution bounds: at most 32 samples, a fixed seed (`42`), and at most 1 training epoch / 20 update steps.
2. Asserts interface correctness, finite outputs, and expected prediction shapes on CPU without GPU, external credentials, or real dataset downloads.
3. Explicitly labels an evaluated naive baseline model (`constant_mean_baseline`).

## Synthetic Fixture Specification
> [!IMPORTANT]
> **DISCLAIMER:** This dataset is an explicitly synthetic, deterministic formula fixture (`target = 2 * x1 + 1`) created solely for offline pipeline verification and regression smoke-testing. It does **NOT** contain real-world forest fire observations, meteorological readings, or environmental burn area measurements.

### Schema
| Field | Type | Description |
|---|---|---|
| `record_id` | String | Unique identifier (`r1`..`r6`) |
| `x1` | Numeric | Scalar feature `[0, 1, 2, 3, 4, 5]` |
| `x2` | Numeric | Constant feature `[1, 1, 1, 1, 1, 1]` |
| `target` | Numeric | Deterministic formula target `2 * x1 + 1` |

### Split Manifest
- **Training:** `["r1", "r2", "r3", "r4"]`
- **Validation:** `["r5"]`
- **Test:** `["r6"]`

## Smoke Run Characteristics
- **Algorithm:** Linear Regression with Stochastic Gradient Descent (SGD) and Constant Mean Baseline.
- **Bounds Checked:**
  - Max samples: $\le 32$ (enforced by `validate_records`).
  - Max epochs: $\le 1$ epoch (1 epoch executed on 4 training rows).
  - Max update steps: $\le 20$ (4 update steps recorded).
- **Outputs:**
  - Training MSE: Finite float.
  - Validation prediction shape: `(1,)`.
  - Test prediction shape: `(1,)`.

## Test Execution
Run the offline unit tests:
```bash
python3 -m unittest discover -s course-labs/r41-t006 -p "test_*.py" -v
```
All 14 tests execute offline in <0.01 seconds using Python standard library only.
