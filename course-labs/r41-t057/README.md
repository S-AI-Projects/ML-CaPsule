# R41-T057: Time-Series Configuration Reproducibility Fixture

This bounded, standard-library-only component verifies serializable
configuration for a forecasting interface. The ten fictional observations
are a **pipeline verification fixture**, not the real bike-sharing demand
dataset and not evidence of real-world or clinical performance.

## Data contract

`fixture.json` is marked `"simulated": true`. It contains ten records with
exactly `record_id`, `timestamp`, and `value`, using IDs `d1` through `d10`,
ten consecutive daily dates, and values `[10, 12, 11, 13, 15, 14, 16, 18, 17,
19]`.

## Configuration and pipeline

`pipeline_config.json` demonstrates the strict, JSON-serializable config:

- schema version: `r41-t057.config.v1`
- seed: `42` (recorded for reproducibility; it does not shuffle time)
- feature order: `["value"]`
- chronological splits: train d1-d6, validation d7-d8, test d9-d10
- algorithm settings: `last_value`, window size 1

The pipeline uses the final training value (`14`) to forecast validation and
test records. Configuration validation rejects unknown fields, unsupported
schema versions, incompatible feature order or algorithm settings, and split
IDs that do not exactly match the chronological partition. Reloading a saved
configuration reproduces both split IDs and forecasts.

## Run and limits

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t057 -p "test_*.py"
```

The checks are offline, CPU-only, deterministic, and use only Python's
standard library. No real dataset is downloaded or used. A real-data extension
must separately document its source, licence, access requirements, and split
design.
