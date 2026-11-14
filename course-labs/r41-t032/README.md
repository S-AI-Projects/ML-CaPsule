# R41-T032: Prevent Split Leakage — Seeds (UCI) – k-means

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`). These synthetic records are strictly for pipeline and split-leakage verification—they are **not** observations from the named real Seeds (UCI) dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

- `record_id`: non-empty unique `str` (e.g. `s1`..`s6`)
- `group_id`: non-empty `str` (e.g. `grp_a`, `grp_b`)
- `x1`: finite scalar `float`
- `x2`: finite scalar `float`
- `expected_cluster`: integer cluster assignment in `{0, 1}`

## Synthetic Clustering Fixture

Six points in 2D space with stable IDs:
- `s1`: (0.0, 0.0) -> Cluster 0
- `s2`: (0.0, 1.0) -> Cluster 0
- `s3`: (1.0, 0.0) -> Cluster 0
- `s4`: (5.0, 5.0) -> Cluster 1
- `s5`: (5.0, 6.0) -> Cluster 1
- `s6`: (6.0, 5.0) -> Cluster 1

Fixed Centroids: `C0 = (0.0, 0.0)` and `C1 = (5.0, 5.0)`.
Nearest-centroid assignments evaluate to `[0, 0, 0, 1, 1, 1]`.

## Deterministic Leakage-Free Splitting

The pipeline creates a deterministic `SplitManifest` satisfying:
1. **Mutual Disjointness:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
2. **Deterministic Reproducibility:** Repeated runs with the same seed yield identical record manifests.
3. **Leakage Detection:** Intentionally overlapping IDs or cross-split group contamination are rejected.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t032 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Bounded offline CPU-only verification component only.
- Any optional real-data extension must separately record source, licence, access requirements, and split design.
