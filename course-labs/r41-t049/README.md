# R41-T049: Avocado prices regression robustness fixture

This bounded contribution checks ID-aligned record-order robustness for a numeric
regression interface in the Avocado prices project context. The six rows are
**simulated data** from the literal deterministic formula `target = 2 * x1 + 1`.
They are not Avocado real-world observations. Results below are pipeline
verification only, not evidence of real-world, market, or clinical performance.

## Data contract

`fixture.json` is explicitly marked `simulated` and `deterministic_formula`.
Every row has exactly `record_id`, `x1`, `x2`, and `target`:

| record_id | x1 | x2 | target | Split |
|---|---:|---:|---:|---|
| r1 | 0 | 1 | 1 | train |
| r2 | 1 | 1 | 3 | train |
| r3 | 2 | 1 | 5 | train |
| r4 | 3 | 1 | 7 | train |
| r5 | 4 | 1 | 9 | validation |
| r6 | 5 | 1 | 11 | test |

Record IDs are unique stable strings. Numeric fields must be finite and booleans
are rejected. Splits are explicit and partition all records exactly once. The
fixture seed is 49. There is no time column and no randomization: record order
may change, but stable IDs must resolve each row independently.

## Transformation and result

The valid numeric transformation is ordinary linear regression using `x1` only.
`x2` remains present in the documented schema and fixture, but is constant and
is therefore not fitted as a redundant coefficient. The train-only fit is
intercept `1.0`, weight `2.0`. Validation `r5` predicts `9.0` versus `9.0`, and
test `r6` predicts `11.0` versus `11.0`. Reversing the record sequence produces
exactly the same pipeline result because IDs, rather than positions, align rows
with splits.

The concrete unsupported cases are: missing fields or extra/missing record
fields, unknown split IDs, duplicate IDs, split partition leakage, non-numeric
or non-finite features, and record shape mismatch. These are rejected with
`RegressionValidationError`; there is no broad fallback or category encoding.

## Limits

This is a bounded robustness fixture, not a claim of broad adversarial
robustness or performance generalization. It has no unseen category/token
because the numeric modality has none. The tested property is specifically
record reordering with stable IDs, plus rejection of missing fields and shape
mismatches. No real Avocado price source, licence, access method, or real-data
split is represented. Any future real-data extension must separately document
those items and use an appropriate time-aware split design.

## Run

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t049 -p "test_*.py"
```

The implementation uses only the Python standard library. It is offline,
CPU-only, fixed-seed as metadata, and bounded to six rows. A failed assertion
causes a non-zero unittest exit. Inputs, expected versus actual results,
environment versions, and mutation evidence are under `work/R41-T049/`.
