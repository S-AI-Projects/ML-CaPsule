# R41-T015: Make Preprocessing Explicit: Glass Identification – k-NN

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic numeric fixture (`fixture.json`). These records are strictly for explicit numeric preprocessing and validation verification—they are **not** observations from the real Glass identification dataset and are **not** evidence of real-world or clinical performance.

## Data Contract & Synthetic Fixture

- **Schema:** `(record_id, group_id, x1, x2, label)`
- **Fixture:** 8 synthetic records (`r1`..`r8`) with $x1 = [0, 1, 2, 3, 4, 5, 6, 7]$, $x2 = 1.0$, and $\text{label} = [0, 0, 0, 0, 1, 1, 1, 1]$.
- **Fixed Stratified Split:**
  - **Train:** `["r1", "r2", "r5", "r6"]` (two 0s, two 1s; $x1 = [0, 1, 4, 5]$, mean = 2.5)
  - **Validation:** `["r3", "r7"]`
  - **Test:** `["r4", "r8"]`

## Explicit Numeric Preprocessing Policy

- **Train-Only Fitting:** Statistics ($\mu, \sigma$) are fitted strictly on training records (`train`).
- **Frozen Transformation:** The public `.transform()` method applies frozen parameters without fitting or adapting to validation or test data.
- **Contamination Rejection:** Missing required features, non-numeric values, or invalid label sets raise explicit `ValueError` exceptions.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t015 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Offline CPU-only bounded verification component only.
- Any optional real-data extension must separately record source, license, access requirements, and feature scaling design.
