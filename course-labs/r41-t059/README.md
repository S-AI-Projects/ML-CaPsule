# R41-T059: Melbourne Daily Minimum-Temperature Robustness Fixture

This is a bounded **pipeline verification fixture** for a time-series
forecasting interface. Its ten fictional observations are not the real
Melbourne daily minimum temperatures dataset and are not evidence of
real-world forecasting performance.

## Data contract

`fixture.json` is marked `"simulated": true` and contains ten records with
exactly `record_id`, `timestamp`, and `value`. The IDs are `d1` through `d10`;
timestamps are ten consecutive dates; values are `[10, 12, 11, 13, 15, 14,
16, 18, 17, 19]`.

## Interface and robustness check

The standard-library-only `pipeline.py` validates the record schema and
timestamps. Its fixed split sorts by timestamp (never randomizes time order):

- Training: d1-d6
- Validation: d7-d8
- Test: d9-d10

The baseline forecasts each validation/test record using the final training
value (14). The valid transformation under test permutes input record order
while retaining stable IDs. Splitting reorders by timestamp, and predictions
are compared after alignment by `record_id`. A missing field is unsupported
and raises `ValueError`. This is a narrow interface check, not a broad
adversarial-robustness claim.

## Run and verify

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t059 -p "test_*.py"
```

The check is offline, CPU-only, deterministic, and uses only the Python
standard library. Seed 42 is used only for the test's reproducible input
permutation; the time-series split itself does not shuffle.

## Limits

No real data is downloaded or used. A real-data extension must separately
document its source, licence, access requirements, and chronological split
design.
