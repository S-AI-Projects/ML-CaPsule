# R41-T037: Record Reproducible Configuration — California housing – linear regression

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`) and serializable configuration (`config.json`). These synthetic records are strictly for pipeline and configuration reproducibility verification—they are **not** observations from the named real California housing dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

- `record_id`: non-empty unique `str` (`r1`..`r6`)
- `x1`: finite scalar `float` in `{0, 1, 2, 3, 4, 5}`
- `x2`: constant feature scalar `float` (`1.0`)
- `target`: deterministic formula $2 \cdot x_1 + 1$
- `split`: split partition in `{"train", "validation", "test"}`

## Reproducible Configuration Contract

The serializable configuration `config.json` enforces:
1. **Schema Versioning:** Fixed schema `1.0.0`; unknown versions or fields are rejected.
2. **Deterministic Partitioning:** Explicit record IDs allocated per split (`train`, `validation`, `test`) with overlap rejection.
3. **Algorithm Settings:** Explicit model family (`linear_regression`) and parameters.
4. **Environment Recording:** Captures runtime dependencies (`Python 3.9.6`, operating system).
5. **Shape Invariance:** Input records missing configured features are rejected before computation.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t037 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Bounded offline CPU-only verification component only.
- Any optional real-data extension must separately record source, licence, access requirements, and split design.
