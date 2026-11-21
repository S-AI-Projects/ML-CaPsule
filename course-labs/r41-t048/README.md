# R41-T048: Medical insurance costs regression error analysis

This bounded contribution demonstrates a row-level error report in the
Medical Insurance Costs project context. The fixture is a six-row
**simulated deterministic formula fixture**, not observations from an
insurance-cost dataset. Its results are pipeline verification only, not
evidence of real-world or clinical performance.

## Data contract and split

`fixture.json` contains exactly six records matching `schema.json`:

| record_id | x1 | x2 | target |
|---|---:|---:|---:|
| r1 | 0 | 1 | 1 |
| r2 | 1 | 1 | 3 |
| r3 | 2 | 1 | 5 |
| r4 | 3 | 1 | 7 |
| r5 | 4 | 1 | 9 |
| r6 | 5 | 1 | 11 |

The target follows `target = 2*x1 + 1`; `x2` is constant. Record IDs are
unique non-empty strings, features and target are finite numbers, and records
may not contain missing or extra fields. A fixed seed (`48`) is metadata; no
random operation is used.

The explicit split is training `r1`–`r4`, validation `r5`, and test `r6`.
Split validation requires every record exactly once, with no overlap.

## Error report

The report compares supplied predictions with each fixture target and
classifies only the residual direction: `no_error`, `overprediction`, or
`underprediction`. It does not infer a causal explanation for a model error.
For testability, the deterministic formula predictions are used for all rows
except `r6`, whose prediction is deliberately increased from `11` to `12`.
The report therefore has one error on `r6` (`overprediction`, absolute error
`1`), no errors in training or validation, and total error count `1`.
Correct formula predictions are labeled `no_error` and omitted from error
counts.

These values verify report construction and aggregation on a deliberately
simple fixture. They are not an evaluation of insurance-cost estimation and
must not be presented as real-world or clinical performance.

## Run checks

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t048 -p "test_*.py"
```

The implementation and tests use only the Python standard library: no
NumPy, scikit-learn, network access, GPU, or external dataset is required.
Tests are offline, CPU-only, and bounded to six records. A failed assertion
makes `unittest` exit non-zero. Inputs, expected and actual results, and
environment versions are recorded under `work/R41-T048/`.

No real-data extension is included. Any future extension must separately
record its source, licence, access requirements, and leakage-aware split
design before reporting dataset-specific results.
