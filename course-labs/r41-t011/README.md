# R41-T011: Validate the Data Contract — Netflix titles – EDA

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`). These synthetic records are strictly for pipeline and data-contract verification—they are **not** observations from the named real Netflix titles dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

- `record_id`: non-empty unique `str`
- `group_id`: non-empty `str`
- `x1`: finite scalar `int` or `float`
- `x2`: finite scalar `int` or `float`
- `label`: binary `int` in `{0, 1}`

## Synthetic Fixture & Stratified Split

- 8 synthetic rows (`rec_00`..`rec_07`) with `x1=[0,1,2,3,4,5,6,7]`, `x2=1`, `label=[0,0,0,0,1,1,1,1]`, `seed=42`.
- `train`: `rec_00, rec_01, rec_04, rec_05` | `validation`: `rec_02, rec_06` | `test`: `rec_03, rec_07`.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t011 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Bounded offline CPU-only verification component only.
- Any optional real-data extension must separately record source, licence, access requirements, and split design.
