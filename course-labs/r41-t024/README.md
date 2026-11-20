# R41-T024 · Loan Approval – Decision Tree: Classification Evaluation

**Task:** R41-T024 · 4 of 10 | **Student:** Sharon Robinson | **Branch:** R41-T024

## Goal

Implement and independently cross-check the evaluation metrics (precision, recall, F1) used for the binary classification fixture. Explicitly define averaging mode and zero-denominator behaviour. All results use a synthetic fixture — not the named real Loan Approval dataset.

---

## Data Contract

> **SYNTHETIC DATA DISCLAIMER:** All rows below are synthetically generated for pipeline verification only. They are not observations from any real Loan Approval dataset and must not be cited as evidence of real-world or clinical performance.

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `record_id` | string | r1–r8 | Stable unique row identifier |
| `group_id` | string | g1–g4 | Stable group for stratified splitting |
| `x1` | integer | 0–7 | Synthetic feature (sequential index) |
| `x2` | integer | 1 (constant) | Synthetic feature (bias-like constant) |
| `label` | integer | 0 or 1 | Binary class (0 = negative, 1 = positive) |
| `split` | string | train/validation/test | Fixed stratified split assignment |

### Fixture Rows

| record_id | group_id | x1 | x2 | label | split |
|-----------|----------|----|----|-------|-------|
| r1 | g1 | 0 | 1 | 0 | train |
| r2 | g1 | 1 | 1 | 0 | train |
| r3 | g2 | 2 | 1 | 0 | validation |
| r4 | g2 | 3 | 1 | 0 | test |
| r5 | g3 | 4 | 1 | 1 | train |
| r6 | g3 | 5 | 1 | 1 | train |
| r7 | g4 | 6 | 1 | 1 | validation |
| r8 | g4 | 7 | 1 | 1 | test |

### Stratified Split Design

| Split | Records | Label-0 count | Label-1 count |
|-------|---------|---------------|---------------|
| Train | r1, r2, r5, r6 | 2 | 2 |
| Validation | r3, r7 | 1 | 1 |
| Test | r4, r8 | 1 | 1 |

No group spans two splits (g1/g2 are label-0 groups; g3/g4 are label-1 groups).

---

## Evaluation Implementation

### Metrics Defined

#### Precision (binary)
$$\text{Precision} = \frac{TP}{TP + FP}$$

Zero-denominator (no predicted positives): controlled by `zero_policy` parameter.

#### Recall (binary)
$$\text{Recall} = \frac{TP}{TP + FN}$$

Zero-denominator (no actual positives): controlled by `zero_policy` parameter.

#### F1 Score
$$\text{F1} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Averaging Modes

| Mode | Description |
|------|-------------|
| `binary` | Positive-class only (default) |
| `macro` | Unweighted mean of per-class metrics |
| `micro` | Aggregate TP/FP/FN across all classes, then compute |

### Zero-Denominator Policy

| Policy | Behaviour | Use case |
|--------|-----------|----------|
| `zero` | Returns `0.0` | Safe default; allows averaging |
| `nan` | Returns `float('nan')` | Flags undefined metric explicitly |
| `one` | Returns `1.0` | Some frameworks treat no predictions as "perfect" |

---

## Canonical Metric Example

**Required verification (from task spec):**

| Input | Values |
|-------|--------|
| truth | `[1, 1, 0, 0]` |
| prediction | `[1, 0, 1, 0]` |

**Confusion matrix:**

| | Predicted 1 | Predicted 0 |
|---|---|---|
| **Actual 1** | TP = 1 | FN = 1 |
| **Actual 0** | FP = 1 | TN = 1 |

**Results:**

| Metric | Formula | Value |
|--------|---------|-------|
| Precision | 1 / (1+1) | **0.5** |
| Recall | 1 / (1+1) | **0.5** |
| F1 | 2×0.5×0.5 / (0.5+0.5) | **0.5** |

---

## Tests

```bash
python3 -m unittest discover -s course-labs/r41-t024 -p "test_*.py" -v
```

| # | Test | Expected |
|---|------|---------|
| 1 | `test_fixture_eight_synthetic_rows` | 8 rows, x1=[0..7], x2=1, label=[0,0,0,0,1,1,1,1] |
| 2 | `test_stratified_split_separation` | train=4, val=2, test=2; each split class-balanced |
| 3 | `test_canonical_precision_recall_half` | precision=0.5, recall=0.5 |
| 4 | `test_confusion_matrix_canonical` | TP=FP=FN=TN=1 |
| 5 | `test_zero_denominator_policy_zero` | precision=0.0 when no positives predicted |
| 6 | `test_zero_denominator_policy_nan` | precision=NaN when no positives predicted |
| 7 | `test_macro_averaging_canonical` | macro precision=0.5, recall=0.5 |
| 8 | `test_perfect_prediction` | precision=recall=f1=1.0 |
| 9 | `test_deliberate_mutation_failure` | mutated expected (0.9) ≠ actual (0.5) |
| 10 | `test_environment_info_recorded` | task_id=R41-T024 present |

Result: **Ran 10 tests … OK**

---

## Limits

- Fixture is a CPU smoke test only — 8 synthetic rows.
- Metrics are implemented from scratch with the Python standard library (no numpy, no scikit-learn).
- Results on this fixture are **not** evidence of performance on any real Loan Approval dataset.

---

## Optional Real-Data Extension

Any real-data extension must separately record:
- **Source:** UCI Loan Approval dataset or equivalent
- **Licence:** Confirm before use
- **Access requirements:** Download link, authentication if any
- **Split design:** Stratified k-fold or time-based as appropriate
- **Data disclaimer:** Results on real data must be reported separately and not conflated with this synthetic fixture

---

## Verification Evidence

See `work/R41-T024/` for:
- `inputs_fixture.json` — fixture copy
- `expected_vs_actual.json` — recorded metric results
- `unittest_output.txt` — full test run output
- `mutation_check.txt` — deliberate-mutation guard output
- `tool_versions.txt` — Python / git / OS versions
