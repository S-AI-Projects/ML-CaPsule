# R41-T026 · Building Energy Efficiency – Regression: Bounded CPU Smoke Run

**Task:** R41-T026 · 6 of 10 | **Student:** Sharon Robinson | **Branch:** `R41-T026` | **Assigned working branch:** `student/e26b07f0819`  
**Repository:** [S-AI-Projects/ML-CaPsule](https://github.com/S-AI-Projects/ML-CaPsule)

---

## Goal

Build a tiny, deterministic, offline CPU-only regression pipeline smoke test for the **Building energy efficiency – regression** teaching component:

- **Bounded resources**: at most `32` samples, fixed seed (`42`), and at most `1` training epoch / `20` update steps — requiring no GPU, credentials, network access, or dataset downloads.
- **Interface verification**: asserts finite outputs and MSE/MAE losses, exact prediction shape `(n_samples,)` with `dtype="float64"`, recorded step count, deterministic reproducibility, and explicit train/validation/test separation — asserting **interface correctness, not useful predictive accuracy**.
- **Explicit model labelling**: includes both a tiny bounded 1-epoch linear SGD smoke model (`BoundedLinearSGDSmokeRegressor`) and an explicitly labelled constant/naive training-mean baseline (`NaiveMeanBaseline`, `is_naive_baseline=True`).

> **SYNTHETIC DATA DISCLAIMER:** All six rows in `fixture.json` are a **deterministic formula fixture** (`target = 2 * x1 + 1`) strictly for **pipeline verification**. They are **not** observations from the UCI Energy Efficiency dataset or any real-world building dataset, and results must **not** be cited as evidence of real-world or clinical performance.

---

## Data Contract

Defined formally in [`schema.json`](schema.json) and instantiated literally in [`fixture.json`](fixture.json).

| Field | Type | Constraint | Description |
|-------|------|------------|-------------|
| `record_id` | `string` | Unique, non-empty | Stable row identifier (`r1`–`r6`) |
| `x1` | `number` | Finite numeric (no `bool`/`NaN`/`Inf`) | Synthetic feature `[0, 1, 2, 3, 4, 5]` |
| `x2` | `number` | Finite numeric (no `bool`/`NaN`/`Inf`) | Constant intercept feature `1` |
| `target` | `number` | Finite numeric (`2 * x1 + 1`) | Deterministic synthetic regression target `[1, 3, 5, 7, 9, 11]` |
| `split` | `string` | `train` \| `validation` \| `test` | Fixed disjoint partition assignment |

### Literal Deterministic Formula Fixture (`fixture.json`, `seed=42`)

| record_id | x1 | x2 | target (`2*x1 + 1`) | split |
|-----------|----|----|---------------------|-------|
| `r1` | `0` | `1` | `1` | `train` |
| `r2` | `1` | `1` | `3` | `train` |
| `r3` | `2` | `1` | `5` | `train` |
| `r4` | `3` | `1` | `7` | `train` |
| `r5` | `4` | `1` | `9` | `validation` |
| `r6` | `5` | `1` | `11` | `test` |

---

## Smoke-Run Results (Synthetic Pipeline Verification Only)

### 1. `BoundedLinearSGDSmokeRegressor` (`seed=42`, `lr=0.05`, `epochs=1`, `step_count=4 <= 20`)

Fitted strictly on training rows `r1..r4` (deterministic shuffle order `[r3, r2, r4, r1]`):
- **Recorded step count**: `4` (within bound `<= 20`); **epochs completed**: `1` (within bound `<= 1`)
- **Step losses (squared error)**: `[25.0, 2.25, 8.41, 0.0036]` (all finite)
- **Learned weights**: `w1 = 2.02`, `w2 = 0.946`

| Split | `record_id`s | Prediction Shape | Dtype | Predictions | Targets | MSE Loss |
|-------|--------------|------------------|-------|-------------|---------|----------|
| `train` | `r1, r2, r3, r4` | `(4,)` | `float64` | `[0.946, 2.966, 4.986, 7.006]` | `[1.0, 3.0, 5.0, 7.0]` | `0.001076` |
| `validation` | `r5` | `(1,)` | `float64` | `[9.026]` | `[9.0]` | `0.000676` |
| `test` | `r6` | `(1,)` | `float64` | `[11.046]` | `[11.0]` | `0.002116` |

### 2. `NaiveMeanBaseline` (`model_kind="constant_naive_baseline"`, `is_naive_baseline=True`)

Explicitly labelled constant/naive baseline predicting the training target mean (`4.0`, computed strictly from `r1..r4`; `step_count=1`, `epochs_completed=1`):

| Split | `record_id`s | Prediction Shape | Dtype | Predictions | Targets | MSE Loss |
|-------|--------------|------------------|-------|-------------|---------|----------|
| `train` | `r1, r2, r3, r4` | `(4,)` | `float64` | `[4.0, 4.0, 4.0, 4.0]` | `[1.0, 3.0, 5.0, 7.0]` | `5.0` |
| `validation` | `r5` | `(1,)` | `float64` | `[4.0]` | `[9.0]` | `25.0` |
| `test` | `r6` | `(1,)` | `float64` | `[4.0]` | `[11.0]` | `49.0` |

---

## Running the Automated Check

The implementation uses only the Python 3 standard library, so the check runs offline and CPU-only with zero external dependencies:

```bash
python3 -m unittest discover -s course-labs/r41-t026 -p "test_*.py" -v
```

### Unit Test Coverage (20 tests)

- `TestDeterministicFormulaFixture` (3): synthetic/formula labelling, literal 6 rows (`target = 2*x1 + 1`), schema fields and smoke bounds
- `TestExplicitTrainValTestSeparation` (2): exact split membership (`r1..r4`, `r5`, `r6`), pairwise disjointness and completeness
- `TestBoundedSGDSmokeModel` (5): epoch/step count bounds, finite step losses and exact weights, exact shapes/dtypes/values, `max_steps` enforcement, fixed-seed determinism
- `TestExplicitConstantNaiveBaseline` (2): explicit `is_naive_baseline=True` labelling, training-only mean `4.0`, exact shapes and MSE losses
- `TestSmokeBoundsAndFailureCases` (7): `> 32` sample rejection, `epochs > 1` / `max_steps > 20` / invalid `lr` rejection, non-train rows in `fit`, overlapping `record_id`, train row leaked into held-out `evaluate`, predict-before-fit error, non-finite/missing/bool/formula violations
- `TestDeliberateMutationGuard` (1): proves altered expected predictions fail assertions

---

## Limits

- Verifies **bounded CPU training/inference interface correctness only** (`<= 32` samples, `<= 1` epoch, `<= 20` steps); low synthetic MSE on a noiseless linear formula (`target = 2 * x1 + 1`) is **not** evidence of predictive accuracy on real building thermal loads.
- Single-row validation (`r5`) and test (`r6`) partitions verify split plumbing and shapes only; they carry no statistical power for model selection or generalization estimates.
- Models single-target scalar regression (`target`), whereas the real UCI Building Energy Efficiency dataset includes two distinct physical responses (heating load and cooling load).

---

## Optional Real-Data Extension Contract

Any future extension using real building energy data **must separately document** its provenance and must never conflate real-data metrics with synthetic smoke-test results:

| Requirement | Specification |
|-------------|---------------|
| **Source** | UCI Machine Learning Repository — Energy Efficiency Dataset (Tsanas & Xifara, 2012) |
| **Licence** | Creative Commons Attribution 4.0 International (CC BY 4.0) — verify upstream before redistribution |
| **Access requirements** | Publicly downloadable via https://archive.ics.uci.edu/dataset/242/energy+efficiency (no paid service or credentials required; keep raw spreadsheets/CSVs out of Git per `.gitignore`) |
| **Split design** | 768 simulated building shapes with 8 features and 2 targets (Heating Load `Y1`, Cooling Load `Y2`); group-aware train/validation/test split by building geometry family to prevent near-duplicate orientation leakage across splits |

---

## Tool Versions

- **Runtime requirement**: Python 3 standard library only (verified offline on Python `3.15.2`, macOS arm64; no GPU, credentials, or external packages required)
