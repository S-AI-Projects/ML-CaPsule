# R41-T038: Explain an Error Case — Mobile price range – classification

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`). These synthetic records are strictly for pipeline and error analysis verification—they are **not** observations from the named real Mobile price range dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

- `record_id`: non-empty unique `str` (`c1`..`c8`)
- `group_id`: category grouping `str` (`grp_low`, `grp_high`)
- `x1`: finite scalar `float` in `{0..7}`
- `x2`: constant feature scalar `float` (`1.0`)
- `label`: binary class integer in `{0, 1}`

## Synthetic Fixture & Stratified Split

Eight balanced synthetic rows:
- Class 0 (`grp_low`): `c1`, `c2`, `c3`, `c4` with $x_1 \in \{0, 1, 2, 3\}$
- Class 1 (`grp_high`): `c5`, `c6`, `c7`, `c8` with $x_1 \in \{4, 5, 6, 7\}$
- Fixed Stratified Split:
  - `train`: `c1`, `c2`, `c5`, `c6` (2 class 0, 2 class 1)
  - `validation`: `c3`, `c7` (1 class 0, 1 class 1)
  - `test`: `c4`, `c8` (1 class 0, 1 class 1)

## Error Analysis Invariants

1. **Deliberate Error Injection:** Record `c7` (true label = 1) is deliberately assigned prediction = 0 to establish a testable error case.
2. **Identification Accuracy:** `c7` is explicitly flagged in the detailed error log with category `false_negative_underestimation`.
3. **Purity of Error Log:** All 7 correct predictions (`c1`..`c6`, `c8`) are excluded from the error list.
4. **Aggregate Alignment:** Aggregate error metrics ($\text{error\_count} = 1$, $\text{total} = 8$, $\text{error\_rate} = 12.5\%$) exactly match the detailed error list length.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t038 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Bounded offline CPU-only verification component only.
- Any optional real-data extension must separately record source, licence, access requirements, and split design.
