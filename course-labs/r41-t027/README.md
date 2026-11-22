# R41-T027 · Banknote Authentication – SVM: Reproducible Configuration

**Task:** R41-T027 · 7 of 10 | **Student:** Sharon Robinson | **Branch:** `R41-T027` | **Assigned working branch:** `student/e26b07f0819`  
**Repository:** [S-AI-Projects/ML-CaPsule](https://github.com/S-AI-Projects/ML-CaPsule)

---

## Goal

Write and validate a **serializable, reproducible configuration** for the **Banknote authentication – SVM** teaching component (`course-labs/r41-t027`):

- Persist `schema_version`, `task_id`, `seed`, `feature_ordering`, `valid_labels`, explicit `split_ids` (`train`/`validation`/`test`), and SVM `algorithm` settings in [`config.json`](config.json).
- Reject **unknown fields**, **unsupported schema versions**, **contradictory settings** (e.g., `kernel="linear"` with non-null `gamma`, `kernel="rbf"` without positive `gamma`, `name="linear_svm"` with non-linear kernel, overlapping `split_ids`), and **incompatible input shapes**.
- Verify that reloading a serialized configuration reconstructs the **exact same splits and predictions**, and record exact **environment versions**.

> **SYNTHETIC DATA DISCLAIMER:** All eight rows in `fixture.json` and all outputs below are **pipeline-verification outputs** from a synthetic classification fixture. They are **not** observations from the UCI Banknote Authentication dataset or any real-world dataset, and must **not** be cited as evidence of real-world or clinical performance.

---

## Data Contract & Serializable Configuration

### Fixture Schema ([`schema.json`](schema.json) / [`fixture.json`](fixture.json))

| record_id | group_id | x1 | x2 | label | Configured Split (`config.json`) |
|-----------|----------|----|----|-------|----------------------------------|
| `r1` | `g1` | `0` | `1` | `0` | `train` |
| `r2` | `g1` | `1` | `1` | `0` | `train` |
| `r3` | `g2` | `2` | `1` | `0` | `validation` |
| `r4` | `g3` | `3` | `1` | `0` | `test` |
| `r5` | `g4` | `4` | `1` | `1` | `train` |
| `r6` | `g4` | `5` | `1` | `1` | `train` |
| `r7` | `g5` | `6` | `1` | `1` | `validation` |
| `r8` | `g6` | `7` | `1` | `1` | `test` |

All three splits are reconstructed strictly from `config.split_ids`, are pairwise disjoint on both `record_id` and `group_id`, and maintain 50/50 class stratification.

### Configuration Contract ([`config.json`](config.json))

- `schema_version`: `"1.0.0"` (unsupported versions raise `SchemaVersionError`)
- `seed`: `42`
- `feature_ordering`: `["x1", "x2"]`
- `valid_labels`: `[0, 1]`
- `split_ids`: `{"train": ["r1", "r2", "r5", "r6"], "validation": ["r3", "r7"], "test": ["r4", "r8"]}`
- `algorithm`: `{"name": "linear_svm", "kernel": "linear", "c": 1.0, "learning_rate": 0.05, "max_iter": 200, "gamma": null}`

Any unrecognized key at any nesting level raises `UnknownConfigFieldError`; contradictory combinations raise `ContradictoryConfigError`; incompatible feature shapes or ordering raise `IncompatibleInputShapeError`.

---

## Pipeline Verification Results (Synthetic Fixture Only)

Running `LinearSVMClassifier` from `config.json` (and verifying identical output after `save()` -> `load()` round-trip):

- **Learned parameters** (fitted strictly on `train` rows `r1, r2, r5, r6`):
  - `weights = [0.5372837221280802, -0.0533138933872721]`
  - `bias = -1.224999999999998`

| Split | `record_id`s | Labels | Decision Scores | Predictions |
|-------|--------------|--------|-----------------|-------------|
| `train` | `r1, r2, r5, r6` | `[0, 0, 1, 1]` | `[-1.2783, -0.7410, +0.8708, +1.4081]` | `[0, 0, 1, 1]` |
| `validation` | `r3, r7` | `[0, 1]` | `[-0.2037, +1.9454]` | `[0, 1]` |
| `test` | `r4, r8` | `[0, 1]` | `[+0.3335, +2.4827]` | `[1, 1]` |

---

## Running the Automated Check

The component uses only the Python 3 standard library, running offline and CPU-only with zero external dependencies:

```bash
python3 -m unittest discover -s course-labs/r41-t027 -p "test_*.py" -v
```

### Unit Test Coverage (18 tests)

- `TestSyntheticFixtureContract` (3): synthetic disclaimer, literal 8 rows & IDs, schema alignment
- `TestConfigSerializationAndReproducibility` (3): JSON file save/load round-trip reproducing identical splits/weights/predictions, expected values, recorded environment versions
- `TestUnknownAndContradictoryConfigRejection` (7): unknown top-level/split/algorithm fields, unsupported `schema_version`, `linear`+`gamma`, `rbf`+`None`/`linear_svm`, overlapping `split_ids`, invalid hyperparameters & duplicate features
- `TestIncompatibleInputShapeAndSplitIntegrity` (4): wrong matrix column width, swapped/missing/extra/non-finite features, group leakage & unknown split ID, predict-before-fit error
- `TestDeliberateMutationGuard` (1): proves altered expected weights trigger `AssertionError`

---

## Limits

- Covers **reproducible configuration serialization, strict validation, and bounded linear SVM verification** only; does not implement full kernel SVM dual optimization or banknote wavelet preprocessing.
- Uses an eight-row synthetic fixture; held-out splits have 2 rows each and provide zero statistical evidence of real-world banknote authentication accuracy.

---

## Optional Real-Data Extension Contract

Any future extension using real banknote data **must separately document** its provenance and distinguish real-data metrics from synthetic pipeline-verification outputs:

| Requirement | Specification |
|-------------|---------------|
| **Source** | UCI Machine Learning Repository — Banknote Authentication Dataset (Volker Lohweg, 2012) |
| **Licence** | Creative Commons Attribution 4.0 International (CC BY 4.0) — verify upstream before redistribution |
| **Access requirements** | Publicly downloadable via https://archive.ics.uci.edu/dataset/267/banknote+authentication (no paid service or credentials required; keep raw data files out of Git per `.gitignore`) |
| **Split design** | 1,372 instances with 4 continuous wavelet-transform features (`variance`, `skewness`, `curtosis`, `entropy`) and binary authenticity label; use stratified train/validation/test split with feature scaling and SVM hyperparameters tuned strictly on training/validation folds |

---

## Tool Versions

- **Runtime requirement**: Python 3 standard library only (verified offline on Python `3.15.2`, CPython, macOS arm64; `schema_version="1.0.0"`)
