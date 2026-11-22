# R41-T050: Indian rainfall saved time-series interface

This bounded contribution checks save/reload behavior for a small time-series
pipeline in the Indian rainfall EDA project context. The ten observations are
**simulated, not Indian rainfall data**. The values are the literal sequence
`[10,12,11,13,15,14,16,18,17,19]` and are pipeline-verification data only, not
evidence of meteorological or climate performance.

## Data contract

`fixture.json` contains exactly these rows:

| record_id | timestamp | value | Split |
|---|---|---:|---|
| d1 | 2025-01-01 | 10 | train |
| d2 | 2025-01-02 | 12 | train |
| d3 | 2025-01-03 | 11 | train |
| d4 | 2025-01-04 | 13 | train |
| d5 | 2025-01-05 | 15 | train |
| d6 | 2025-01-06 | 14 | train |
| d7 | 2025-01-07 | 16 | validation |
| d8 | 2025-01-08 | 18 | validation |
| d9 | 2025-01-09 | 17 | test |
| d10 | 2025-01-10 | 19 | test |

The schema requires exactly `record_id`, `timestamp`, and `value`. IDs are unique
non-empty strings, timestamps use `YYYY-MM-DD`, and values are finite numbers.
The split IDs are fixed, consecutive, and never shuffled.

## Saved parameters and prediction

`fit_configuration` computes a training-only median. The sorted training values
are `[10,11,12,13,14,15]`, so the saved `forecast_value` is `12.5`. The JSON
configuration includes schema version `1`, model type `training_median`, seed
`50`, feature schema, and forecast value. No NumPy, pickle, credentials, network,
or training records are required for inference.

`save_config` and `load_config` use JSON only. `predict(record, config)` validates
the exact prediction record fields and the loaded configuration before returning
the saved forecast. `tolerance = 1e-12` applies to before/after reload agreement.

## Failures and limits

Unsupported schema versions, unknown model types, seed mismatches, feature-schema
mismatch, malformed JSON object fields, non-finite forecast values, missing prediction
fields, and invalid timestamps/values are rejected clearly. Unknown split IDs,
duplicate IDs, split partition overlap, and shuffled or reversed time IDs are also
rejected. This is not an unseen-category test and does not claim broad adversarial
robustness.

This median baseline is a small saved-interface verification, not a rainfall
forecast model. No real dataset, license, source, or access requirements are
asserted. Any real-data extension must separately document the source, licence,
access requirements, and leakage-safe chronological split design.

## Run

```bash
python3 -m unittest discover -s course-labs/r41-t050 -p "test_*.py"
```

The implementation and tests use only the Python standard library. It is offline,
CPU-only, and bounded to ten rows. A failed assertion produces a non-zero unittest
exit. Inputs, expected versus actual results, tool versions, and mutation evidence
are recorded under `work/R41-T050/`.
