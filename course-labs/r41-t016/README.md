# R41-T016: Add a Bounded CPU Smoke Run: Bank Marketing (UCI) – Logistic Regression

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic deterministic formula fixture (`fixture.json`). These records verify pipeline smoke execution, interface correctness, and bounded training loops—they are **not** observations from the real Bank marketing dataset and are **not** evidence of real-world or clinical performance.

## Data Contract & Synthetic Fixture

- **Schema:** `(record_id, x1, x2, target)`
- **Deterministic Formula:** $\text{target} = 2 \cdot x1 + 1$ with $x1 = [0, 1, 2, 3, 4, 5]$ and $x2 = 1.0$.
- **Split Design:**
  - **Train:** `["r1", "r2", "r3", "r4"]`
  - **Validation:** `["r5"]`
  - **Test:** `["r6"]`

## Bounded CPU Smoke Run Specifications

- **Sample Bound:** 6 samples (strictly $\le 32$).
- **Training Bound:** 1 single epoch sweep / 4 update steps (strictly $\le 20$ update steps).
- **Hardware Constraints:** 100% offline CPU-only computation with zero GPU, external network, or credential dependencies.
- **Validation Goal:** Interface verification, finite gradient loss progression, and non-empty output shapes.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t016 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- This is an offline bounded smoke run component only, asserting interface correctness rather than predictive utility.
- Any optional real-data extension must separately record source, license, access requirements, and split design.
