# R41-T025 · Zoo Animals – Decision Tree: Explicit Numeric Preprocessing

**Task:** R41-T025 · 5 of 10 | **Student:** Sharon Robinson | **Branch:** `R41-T025` | **Assigned working branch:** `student/e26b07f0819`  
**Repository:** [S-AI-Projects/ML-CaPsule](https://github.com/S-AI-Projects/ML-CaPsule)

---

## Goal

Implement and verify a bounded numeric preprocessing contract for the **Zoo animals – decision tree** teaching component:

- Fit numeric standardization **on training rows only**.
- Expose an immutable **transform-only public interface** (`FittedNumericPreprocessor`) that has no `fit`, `partial_fit`, or `fit_transform` method, making fitting on held-out input structurally impossible.
- Explicitly validate **feature order**, **label set**, and **missing-value policy**.
- Reject **deliberate contamination fixtures** (non-train rows passed to fitting, overlapping `record_id` or `group_id` across splits, or training rows/groups leaked into held-out transform calls).

> **SYNTHETIC DATA DISCLAIMER:** All eight rows in `fixture.json` and all numerical results below are **pipeline-verification outputs** from a synthetic fixture. They are **not** observations from the UCI Zoo dataset or any real-world dataset, and must **not** be cited as evidence of real-world or clinical performance.

---

## Data Contract

Defined formally in [`schema.json`](schema.json) and instantiated literally in [`fixture.json`](fixture.json).

| Field | Type | Constraint | Description |
|-------|------|------------|-------------|
| `record_id` | `string` | Unique, non-empty | Stable row identifier (`r1`–`r8`) |
| `group_id` | `string` | Non-empty, split-isolated | Stable group identifier (`g1`–`g6`), disjoint across splits |
| `x1` | `number` | Finite numeric (no `bool`/`NaN`/`Inf`) | Synthetic numeric feature `[0, 1, 2, 3, 4, 5, 6, 7]` |
| `x2` | `number` | Finite numeric (no `bool`/`NaN`/`Inf`) | Constant synthetic feature `1` (tests zero-variance safety) |
| `label` | `integer` | Strict membership in `{0, 1}` (no `bool`) | Binary synthetic class label `[0, 0, 0, 0, 1, 1, 1, 1]` |
| `split` | `string` | `train` \| `validation` \| `test` | Fixed stratified, group-isolated partition |

### Literal Synthetic Fixture (`fixture.json`, `seed=42`)

| record_id | group_id | x1 | x2 | label | split |
|-----------|----------|----|----|-------|-------|
| `r1` | `g1` | `0` | `1` | `0` | `train` |
| `r2` | `g1` | `1` | `1` | `0` | `train` |
| `r3` | `g2` | `2` | `1` | `0` | `validation` |
| `r4` | `g3` | `3` | `1` | `0` | `test` |
| `r5` | `g4` | `4` | `1` | `1` | `train` |
| `r6` | `g4` | `5` | `1` | `1` | `train` |
| `r7` | `g5` | `6` | `1` | `1` | `validation` |
| `r8` | `g6` | `7` | `1` | `1` | `test` |

### Fixed Stratified Split Design

| Partition | `record_id`s | `group_id`s | Rows | Class 0 | Class 1 | Used for Fitting? |
|-----------|--------------|-------------|------|---------|---------|-------------------|
| `train` | `r1, r2, r5, r6` | `g1, g4` | 4 | 2 | 2 | **Yes (only)** |
| `validation` | `r3, r7` | `g2, g5` | 2 | 1 | 1 | **No (transform only)** |
| `test` | `r4, r8` | `g3, g6` | 2 | 1 | 1 | **No (transform only)** |

All three partitions are pairwise disjoint on both `record_id` and `group_id` and maintain exact 50/50 class stratification.

---

## Preprocessing Contract (`pipeline.py`)

1. **Training-only fitting (`fit_numeric_preprocessor`)**:
   - Accepts only records where `split == "train"` (when `split` is present); passing `validation` or `test` rows raises `ContaminationError`.
   - Computes per-feature mean and population standard deviation (`ddof=0`) strictly from training rows. Zero-variance features (`x2 = 1`) use safe `scale = 1.0` (matching `sklearn.preprocessing.StandardScaler`), preventing division by zero.
2. **Transform-only public interface (`FittedNumericPreprocessor`)**:
   - Implemented as a `@dataclass(frozen=True)` with immutable tuple statistics (`means`, `stds`, `scales`).
   - Exposes only `transform(...)` and `transform_split(...)`; no `fit`, `partial_fit`, or `fit_transform` methods exist.
   - Returns an immutable `TransformedMatrix` with explicit `shape`, `dtype="float64"`, `feature_names=("x1", "x2")`, `record_ids`, and finite IEEE 754 `float64` values.
3. **Contract validations**:
   - **Feature order** (`FeatureOrderError`): rejects swapped order (`x2` before `x1`), missing features, extra features, or mismatched explicit `feature_order`.
   - **Label set** (`InvalidLabelError`): rejects integers outside `{0, 1}`, booleans (`True`/`False`), floats, strings, and `None`.
   - **Missing-value policy** (`InvalidPolicyError` / `MissingValueError`): supports `MissingValuePolicy.REJECT` (default, rejects `None`, `NaN`, `±Inf`, empty strings, and `bool`) and explicit `MissingValuePolicy.IMPUTE_TRAIN_MEAN` (imputes strictly with the frozen training mean; rejects all-missing training columns).
   - **Contamination rejection** (`ContaminationError`): rejects overlapping `record_id` or `group_id` across partitions and blocks training rows/groups passed into held-out `validation`/`test` transform calls.

---

## Pipeline Verification Results (Synthetic Fixture Only)

Fitted strictly on training rows `r1, r2, r5, r6` (`x1 = [0, 1, 4, 5]`, `x2 = [1, 1, 1, 1]`):

- **Training means**: `x1 = 2.5`, `x2 = 1.0` (full-dataset `x1` mean would be `3.5`, confirming zero held-out leakage)
- **Training population stds (`ddof=0`)**: `x1 = sqrt(4.25) ≈ 2.061552812809`, `x2 = 0.0`
- **Safe scales**: `x1 = 2.061552812809`, `x2 = 1.0`

| Split | Shape | Dtype | `record_id` | `x1_scaled` | `x2_scaled` |
|-------|-------|-------|-------------|-------------|-------------|
| `train` | `(4, 2)` | `float64` | `r1` (`x1=0`) | `-1.212678125182` | `0.0` |
| `train` | `(4, 2)` | `float64` | `r2` (`x1=1`) | `-0.727606875109` | `0.0` |
| `train` | `(4, 2)` | `float64` | `r5` (`x1=4`) | `+0.727606875109` | `0.0` |
| `train` | `(4, 2)` | `float64` | `r6` (`x1=5`) | `+1.212678125182` | `0.0` |
| `validation` | `(2, 2)` | `float64` | `r3` (`x1=2`) | `-0.242535625036` | `0.0` |
| `validation` | `(2, 2)` | `float64` | `r7` (`x1=6`) | `+1.697749375254` | `0.0` |
| `test` | `(2, 2)` | `float64` | `r4` (`x1=3`) | `+0.242535625036` | `0.0` |
| `test` | `(2, 2)` | `float64` | `r8` (`x1=7`) | `+2.182820625327` | `0.0` |

Values match `sklearn.preprocessing.StandardScaler` to within `1e-15` (IEEE 754 `float64`).

---

## Running the Automated Check

The implementation uses only the Python 3 standard library, so the check runs offline and CPU-only with no external dependencies required:

```bash
python3 -m unittest discover -s course-labs/r41-t025 -p "test_*.py" -v
```

### Unit Test Coverage (28 tests)

- `TestSyntheticFixtureContract` (4): synthetic labelling, literal 8 rows, stable IDs, schema alignment
- `TestStratifiedTrainValTestSplit` (3): explicit 3-way split, disjoint record/group IDs, balanced stratification
- `TestTrainingOnlyFitAndTransformOutputs` (5): training-only statistics, `(n, 2)` shapes & `float64` dtype, exact transformed values, standardized train moments vs non-zero held-out means, determinism
- `TestTransformInterfaceCannotFit` (3): absence of `fit`/`partial_fit`/`fit_transform`, frozen immutability, invariance under extreme held-out inputs
- `TestDeliberateContaminationRejection` (6): non-train rows in `fit`, overlapping `record_id`, overlapping `group_id`, held-out group overlap, training rows/groups smuggled into held-out `transform`
- `TestFeatureOrderLabelAndMissingValueContracts` (6): swapped/missing/extra feature columns, invalid labels (including `bool`), invalid policies, default missing/non-finite rejection, training-mean-only imputation
- `TestDeliberateMutationGuard` (1): confirms altered expected outputs raise `AssertionError`

---

## Limits

- Covers the **bounded numeric preprocessing stage** only; it does not train or tune a full multi-class Zoo decision-tree model.
- Uses an 8-row synthetic fixture (`x1=[0..7]`, `x2=1`) strictly for contract and leakage verification; held-out splits have 2 rows each and carry zero statistical power for real-world generalization claims.
- Assumes numeric features; categorical or binary-attribute encoding for real animal traits is out of scope for this component.

---

## Optional Real-Data Extension Contract

Any future extension that evaluates a decision tree on real animal classification data **must separately document** its dataset provenance and must never conflate real-data metrics with synthetic pipeline-verification outputs:

| Requirement | Specification |
|-------------|---------------|
| **Source** | UCI Machine Learning Repository — Zoo Dataset (Richard Forsyth, 1990) |
| **Licence** | Creative Commons Attribution 4.0 International (CC BY 4.0) — verify upstream before redistribution |
| **Access requirements** | Publicly downloadable via https://archive.ics.uci.edu/dataset/111/zoo (no paid service or credentials required; keep raw CSVs out of Git per `.gitignore`) |
| **Split design** | Multi-class (7 animal types, 101 instances); requires stratified train/validation/test split (or nested stratified cross-validation given small minority classes such as amphibians/reptiles) with preprocessing fitted strictly inside each training fold |

---

## Tool Versions

- **Runtime requirement**: Python 3 standard library only (verified on Python `3.15.2`, macOS arm64)
- **Optional cross-check environment**: `numpy==2.5.5`, `scikit-learn==1.9.1` (used offline to confirm `atol=1e-15` agreement with `StandardScaler`)
