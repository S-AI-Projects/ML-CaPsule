# R41-T017: Record Reproducible Configuration: Olivetti Faces – PCA Eigenfaces

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic collinear points fixture (`fixture.json`). These records verify serializable configuration schema enforcement, 1D projection, sign indeterminacy handling, and exact numerical reconstruction—they are **not** observations from the real Olivetti faces dataset and are **not** evidence of real-world or clinical performance.

## Data Contract & Synthetic Fixture

- **Schema:** `(record_id, x1, x2)`
- **Points:** 4 centered collinear points: `(-2, -2)`, `(-1, -1)`, `(1, 1)`, `(2, 2)`.
- **Projection Axis:** Unit vector $v = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)$.

## Reproducible Configuration & Projection Guarantees

- **Serializable Config:** Enforces `schema_version`, `seed`, `feature_ordering`, `split_ids`, and `algorithm_settings`.
- **Strict Validation:** Unrecognized top-level fields, missing keys, or contradictory algorithm settings (`n_components != 1`) trigger an explicit `ValueError`.
- **Sign Invariance:** Reconstruction accuracy is mathematically invariant to component sign reflection ($\pm v$).
- **Tolerance:** Reconstructs points with maximum error strictly $\le 10^{-5}$.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t017 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- This is an offline bounded verification component for reproducible configuration schemas.
- Any optional real-data extension must separately record source, license, access requirements, and split design.
