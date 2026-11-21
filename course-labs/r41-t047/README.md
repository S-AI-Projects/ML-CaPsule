# R41-T047: Breast Cancer Wisconsin feature-selection configuration

This bounded contribution demonstrates a reproducible classification pipeline
and feature-selection configuration in the Breast Cancer Wisconsin project
context. Its eight rows are **synthetic, simulated data**, not observations
from the named real dataset. Results below are pipeline verification only, not
evidence of real-world or clinical performance.

## Data contract

`fixture.json` is marked `simulated` and contains exactly these eight rows:

| record_id | group_id | x1 | x2 | label |
|---|---|---:|---:|---:|
| bcw-syn-001 | bcw-group-001 | 0 | 1 | 0 |
| bcw-syn-002 | bcw-group-002 | 1 | 1 | 0 |
| bcw-syn-003 | bcw-group-003 | 2 | 1 | 0 |
| bcw-syn-004 | bcw-group-004 | 3 | 1 | 0 |
| bcw-syn-005 | bcw-group-005 | 4 | 1 | 1 |
| bcw-syn-006 | bcw-group-006 | 5 | 1 | 1 |
| bcw-syn-007 | bcw-group-007 | 6 | 1 | 1 |
| bcw-syn-008 | bcw-group-008 | 7 | 1 | 1 |

`schema.json` defines the record contract: non-empty unique string IDs,
finite numeric `x1` and `x2`, and integer label `0` or `1` (booleans are not
accepted). Both record and group IDs are stable fixture identifiers; each row
has its own group. Extra or missing record fields are rejected.

## Serialized configuration and split

`config.json` is versioned (`schema_version: 1`) and records seed `47`, feature
order `x1`, `x2`, exact row IDs, and all supported algorithm settings. The
explicit split is stratified and group-disjoint:

| Split | Record IDs | Label counts |
|---|---|---|
| Train | bcw-syn-001, bcw-syn-004, bcw-syn-005, bcw-syn-008 | 2 / 2 |
| Validation | bcw-syn-002, bcw-syn-006 | 1 / 1 |
| Test | bcw-syn-003, bcw-syn-007 | 1 / 1 |

The seed is retained as experiment metadata; no random operation is needed
because the reviewed split IDs are explicitly stored. The training-only
selector ranks features by absolute class-mean gap and keeps the configured
number. A nearest-centroid classifier then uses Euclidean distance on selected
features, breaking equal-distance ties toward the lower label. Unsupported,
unknown, or contradictory configuration fields, split leakage, and
incompatible feature shapes are rejected.

The checked pipeline selects `x1`; its training centroids are `1.5` for label
0 and `5.5` for label 1. Validation and test predictions are both `[0, 1]`
against labels `[0, 1]`. These perfect scores arise from the deliberately
separable toy fixture and are **pipeline verification only**. They do not
measure feature-selection quality on Breast Cancer Wisconsin data and must not
be presented as real-world or clinical performance.

## Run checks

From the repository root, using Python 3:

```bash
python3 -m unittest discover -s course-labs/r41-t047 -p "test_*.py"
```

The implementation and tests use only the Python standard library: no
NumPy, scikit-learn, network access, GPU, or external dataset is required.
Tests are offline, CPU-only, and bounded to eight fixture rows. A failed
assertion makes `unittest` exit non-zero. Inputs, expected and actual results,
and environment versions are recorded under `work/R41-T047/`.

No real-data extension is included. Any future extension must separately
record the actual data source, its licence, access requirements, and a
leakage-aware split design before reporting dataset-specific results.
