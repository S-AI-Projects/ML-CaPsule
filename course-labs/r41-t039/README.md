# R41-T039: Test an Input Perturbation — Wine origin (UCI) – LDA

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`). These synthetic records are strictly for pipeline and input perturbation robustness verification—they are **not** observations from the named real Wine origin (UCI) dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

- `record_id`: non-empty unique `str` (`p1`..`p6`)
- `x1`: finite scalar `float`
- `x2`: finite scalar `float`
- `expected_cluster`: integer cluster assignment in `{0, 1}`

## Synthetic Fixture

Six points in 2D space with stable IDs:
- `p1`: (0.0, 0.0) -> Cluster 0
- `p2`: (0.0, 1.0) -> Cluster 0
- `p3`: (1.0, 0.0) -> Cluster 0
- `p4`: (5.0, 5.0) -> Cluster 1
- `p5`: (5.0, 6.0) -> Cluster 1
- `p6`: (6.0, 5.0) -> Cluster 1

Fixed Centroids: `C0 = (0.0, 0.0)` and `C1 = (5.0, 5.0)`.
Nearest-centroid assignments evaluate to `[0, 0, 0, 1, 1, 1]`.

## Robustness & Perturbation Properties Tested

1. **Valid Transformation (Row-Order Invariance):** Permuting the order of input rows preserves identical cluster assignments for all records after ID alignment. Time order is not randomized.
2. **Unsupported Input (Missing Field):** Omitting a required feature (`x2`) is rejected with an explicit `ValueError`.
3. **Unsupported Input (Shape Mismatch):** Passing 3D centroids to 2D records triggers dimensional validation failure with a clear `ValueError`.
4. **Scope Limitation:** The tests assert scoped interface robustness; they do not claim broad adversarial robustness.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t039 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Bounded offline CPU-only verification component only.
- Any optional real-data extension must separately record source, licence, access requirements, and split design.
