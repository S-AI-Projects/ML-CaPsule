# R41-T030 · Global CO2 Emissions – Trend Forecast: Saved-Model Interface

**Task:** R41-T030 · 10 of 10 | **Student:** Sharon Robinson | **Branch:** `R41-T030` | **Assigned working branch:** `student/e26b07f0819`  
**Repository:** [S-AI-Projects/ML-CaPsule](https://github.com/S-AI-Projects/ML-CaPsule)

---

## Goal

Define a safe, self-contained **save/reload and inference interface** for the **Global CO2 emissions – trend forecast** teaching component (`course-labs/r41-t030`):

- Fit a deterministic closed-form OLS linear trend strictly on chronological training observations (`d1..d6`, **no shuffling**).
- Save and reload model configuration and learned parameters strictly in **JSON** ([`saved_model.json`](saved_model.json)); **never load an untrusted pickle** (`.pkl`/`.pickle` formats are explicitly rejected with `UnsafeArtifactFormatError`).
- Expose `predict_with_artifact(artifact, inference_records)` with strict schema and version validation, requiring **no training data and no credentials** at inference time, and asserting before/after reload prediction agreement within `RELOAD_TOLERANCE = 1e-12`.

> **SYNTHETIC DATA DISCLAIMER:** All ten daily values (`d1..d10`) in `fixture.json` are a **synthetic pipeline-verification fixture** — NOT real Global CO2 emissions observations and NOT evidence of real-world or clinical performance.

---

## Data Contract

Defined in [`schema.json`](schema.json) and instantiated literally in [`fixture.json`](fixture.json).

### Ten Consecutive Daily Observations (No Shuffling)

| record_id | timestamp | value | Chronological Split |
|-----------|-----------|-------|---------------------|
| `d1` | `2026-01-01` | `10` | `train` |
| `d2` | `2026-01-02` | `12` | `train` |
| `d3` | `2026-01-03` | `11` | `train` |
| `d4` | `2026-01-04` | `13` | `train` |
| `d5` | `2026-01-05` | `15` | `train` |
| `d6` | `2026-01-06` | `14` | `train` |
| `d7` | `2026-01-07` | `16` | `validation` |
| `d8` | `2026-01-08` | `18` | `validation` |
| `d9` | `2026-01-09` | `17` | `test` |
| `d10` | `2026-01-10` | `19` | `test` |

---

## Pipeline Verification Results (Synthetic Fixture Only)

Fitted strictly on training rows `d1..d6` (`t = 0..5`):
- **Slope**: `31 / 35 = 0.8857142857142857`
- **Intercept**: `72 / 7 = 10.285714285714286`
- **Reload tolerance**: `1e-12`; observed `max_reload_abs_diff = 0.0`

| Split | record_id | timestamp | Actual Value | Before Reload | After Reload |
|-------|-----------|-----------|--------------|---------------|--------------|
| `validation` | `d7` | `2026-01-07` | `16.0` | `15.600000000000` | `15.600000000000` |
| `validation` | `d8` | `2026-01-08` | `18.0` | `16.485714285714` | `16.485714285714` |
| `test` | `d9` | `2026-01-09` | `17.0` | `17.371428571429` | `17.371428571429` |
| `test` | `d10` | `2026-01-10` | `19.0` | `18.257142857143` | `18.257142857143` |

---

## Running the Automated Check

Uses only the Python 3 standard library (offline, CPU-only, zero external dependencies):

```bash
python3 -m unittest discover -s course-labs/r41-t030 -p "test_*.py" -v
```

### Unit Test Coverage (12 tests)

- `TestChronologicalFixtureContract` (3): synthetic & no-shuffling notices, literal 10 observations `d1..d10`, strict 6/2/2 chronological split
- `TestSavedArtifactReloadAndInference` (3): analytical OLS train-only fit, before/after JSON reload equivalence within `1e-12` with zero embedded training data/credentials, standalone inference without target values
- `TestUnsafePickleSchemaAndVersionFailures` (5): `.pkl`/`.pickle` rejection, `schema_version` mismatch rejection, unknown/non-JSON artifact rejection, shuffled/non-consecutive timestamp rejection, invalid inference schema rejection
- `TestDeliberateMutationGuard` (1): proves altered expected forecasts fail assertions

---

## Limits

- Uses a linear OLS trend on 10 synthetic daily integers purely to verify JSON artifact serialization, schema/version validation, and chronological splitting; does not capture seasonality, structural breaks, policy shifts, or nonlinear carbon-cycle dynamics.
- Validation and test horizons have 2 points each and carry zero statistical power for real climate or emissions forecasting claims.

---

## Optional Real-Data Extension Contract

Any future extension forecasting real global CO2 emissions **must separately document** its provenance and distinguish real-data metrics from synthetic pipeline-verification outputs:

| Requirement | Specification |
|-------------|---------------|
| **Source** | Global Carbon Project / Our World in Data CO2 and Greenhouse Gas Emissions dataset |
| **Licence** | CC BY 4.0 (verify upstream attribution and version licence before redistribution) |
| **Access requirements** | Publicly available via https://github.com/owid/co2-data (no paid service or credentials required; keep raw CSVs out of Git per `.gitignore`) |
| **Split design** | Multi-decade annual/monthly country and global time series; use strictly chronological expanding-window or rolling-origin train/validation/test backtesting (never random shuffling) with feature/trend fitting strictly inside each historical window |

---

## Tool Versions

- **Runtime requirement**: Python 3 standard library only (verified offline on Python `3.15.2`, macOS arm64; JSON-only artifact serialization, no pickle)
