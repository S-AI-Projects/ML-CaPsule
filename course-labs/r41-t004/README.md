# R41-T004: Evaluation Metric Verification (Pokémon Stats)

## Objective

Independently implement and mathematically cross-check evaluation metrics for a classification and regression fixture, explicitly formalizing averaging conventions, zero-denominator behaviors, input validation, and stratified train/validation/test split isolation.

> **DISCLAIMER:**  
> The eight records in this project are strictly simulated synthetic verification data. They do **NOT** represent real Pokémon combat stats, legendary status observations, or actual game data, and must not be used to claim real-world model accuracy.

---

## Data Schema & Synthetic Fixture

Each record is a dictionary conforming to the following data contract:

| Field Name | Type | Constraints | Description |
|---|---|---|---|
| `record_id` | `str` | Required, non-empty, unique string (`rec_001`..`rec_008`) | Stable record identifier |
| `group_id` | `str` | Required, non-empty string (`grp_01`..`grp_04`) | Group identifier for isolation |
| `x1` | `int` or `float` | Required, numeric scalar (`0` through `7`) | First simulated numerical feature |
| `x2` | `int` or `float` | Required, numeric scalar (`1` constant) | Second simulated numerical feature |
| `label` | `int` | Required, binary integer: `0` or `1` | Legendary classification label |

### The Eight Canonical Synthetic Rows

| Record ID | Group ID | x1 | x2 | Label | Description |
|---|---|---|---|---|---|
| `rec_001` | `grp_01` | 0 | 1 | 0 | Non-legendary |
| `rec_002` | `grp_01` | 1 | 1 | 0 | Non-legendary |
| `rec_003` | `grp_02` | 2 | 1 | 0 | Non-legendary |
| `rec_004` | `grp_02` | 3 | 1 | 0 | Non-legendary |
| `rec_005` | `grp_03` | 4 | 1 | 1 | Legendary |
| `rec_006` | `grp_03` | 5 | 1 | 1 | Legendary |
| `rec_007` | `grp_04` | 6 | 1 | 1 | Legendary |
| `rec_008` | `grp_04` | 7 | 1 | 1 | Legendary |

---

## Metric Definitions and Verification

### 1. Classification Metrics (Precision, Recall, F1, Accuracy)

For binary classification with designated positive class $y = 1$:

- **Confusion Matrix:**
  - True Positives ($TP$): $y_{true} = 1 \land y_{pred} = 1$
  - False Positives ($FP$): $y_{true} = 0 \land y_{pred} = 1$
  - False Negatives ($FN$): $y_{true} = 1 \land y_{pred} = 0$
  - True Negatives ($TN$): $y_{true} = 0 \land y_{pred} = 0$

- **Formulas:**
  $$\text{Precision} = \frac{TP}{TP + FP}$$
  $$\text{Recall} = \frac{TP}{TP + FN}$$
  $$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  $$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Issue #4 Concrete Benchmark:**
  - $\text{Truth} = [1, 1, 0, 0]$
  - $\text{Prediction} = [1, 0, 1, 0]$
  - Contingency:
    - Index 0: $y_t=1, y_p=1 \implies TP$
    - Index 1: $y_t=1, y_p=0 \implies FN$
    - Index 2: $y_t=0, y_p=1 \implies FP$
    - Index 3: $y_t=0, y_p=0 \implies TN$
    - $TP=1, FP=1, FN=1, TN=1$
  - Results:
    - **Precision:** $\frac{1}{1 + 1} = \mathbf{0.5}$
    - **Recall:** $\frac{1}{1 + 1} = \mathbf{0.5}$
    - **F1 Score:** $\frac{2 \times 0.25}{1.0} = \mathbf{0.5}$
    - **Accuracy:** $\frac{1 + 1}{4} = \mathbf{0.5}$

### 2. Averaging Convention
- The default averaging convention is **binary**, evaluating performance with respect to the designated positive class (`pos_label=1`).
- For macro-averaging across both classes $0$ and $1$, unweighted per-class precision is:
  $$\text{Precision}_{\text{class 1}} = 0.5, \quad \text{Precision}_{\text{class 0}} = \frac{TN}{TN + FN} = \frac{1}{1 + 1} = 0.5$$
  Yielding $\text{Macro Precision} = \mathbf{0.5}$ and $\text{Macro Recall} = \mathbf{0.5}$.

### 3. Zero-Denominator Handling
- If $TP + FP = 0$ (no positive predictions made), precision is undefined. The parameter `zero_division=0.0` (standard scikit-learn convention) returns `0.0`.
- If $TP + FN = 0$ (no positive instances in ground truth), recall is undefined and returns `zero_division=0.0`.
- Configurable `zero_division` parameter allows deterministic custom fallback (e.g. `1.0`).

### 4. Regression Metric (Mean Absolute Error)

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_{true, i} - y_{pred, i}|$$

- **Issue #4 Concrete Benchmark:**
  - $\text{Truth} = [1, 2, 3]$
  - $\text{Prediction} = [1, 3, 2]$
  - Residuals: $|1 - 1| + |2 - 3| + |3 - 2| = 0 + 1 + 1 = 2$
  - Sample size: $N = 3$
  - **Expected MAE:** $\frac{2}{3} \approx \mathbf{0.66666667}$

---

## Stratified Split Strategy

To guarantee reproducible train/validation/test separation without data leakage:

| Split | Assigned Records | Labels | Count | Class Ratio (0 : 1) |
|---|---|---|---:|---|
| **Train** | `rec_001`, `rec_002`, `rec_005`, `rec_006` | `[0, 0, 1, 1]` | 4 | 2 : 2 (50% / 50%) |
| **Validation** | `rec_003`, `rec_007` | `[0, 1]` | 2 | 1 : 1 (50% / 50%) |
| **Test** | `rec_004`, `rec_008` | `[0, 1]` | 2 | 1 : 1 (50% / 50%) |

- Splits are strictly disjoint: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- Complete partition: all 8 records are assigned exactly once.

---

## How to Run Tests

The test suite runs offline using Python's standard library (`unittest`, `math`, `dataclasses`):

```bash
python3 -m unittest discover -s course-labs/r41-t004 -p "test_*.py" -v
```

---

## Scope & Limitations

- **Bounded Metric Task:** This component verifies evaluation metric calculations and input contracts. It does not train full classification or regression models on Pokémon stats.
- **Offline & CPU-Only:** Does not depend on scikit-learn, NumPy, or GPU compute.
- **Synthetic Scope:** Verifies numerical implementation correctness only.
