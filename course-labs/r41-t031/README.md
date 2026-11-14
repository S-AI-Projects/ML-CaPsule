# R41-T031: Validate the Data Contract — sklearn digits – PCA + SVM

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`). These synthetic records are strictly for pipeline and data-contract verification—they are **not** observations from the named real sklearn digits dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

- `record_id`: non-empty unique `str`
- `x1`: finite scalar `float` or `int`
- `x2`: finite scalar `float` or `int`

## Synthetic Fixture

Four centered collinear points in 2D space:
- `p1`: (-2.0, -2.0)
- `p2`: (-1.0, -1.0)
- `p3`: (1.0, 1.0)
- `p4`: (2.0, 2.0)

## Mathematical Verification

Projection onto unit vector $u = (1/\sqrt{2}, 1/\sqrt{2})$ followed by reconstruction recovers each collinear point within $10^{-6}$ numerical tolerance:
$$\hat{x} = (x \cdot u) u$$
The reconstruction is also invariant to sign reversal of the learned component ($u \to -u$).

## Validation Guarantees

The schema validator `validate_records()` checks for:
- Missing `record_id` or non-string/whitespace values
- Duplicate `record_id` across rows without dropping records
- Missing or non-numeric/tensor features (`x1`, `x2`)
- Non-finite numeric values (`NaN`, `Inf`)

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t031 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Bounded offline CPU-only verification component only.
- Any optional real-data extension must separately record source, licence, access requirements, and split design.
