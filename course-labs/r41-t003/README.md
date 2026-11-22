# R41-T003: SMS Spam Collection — Majority-Class Baseline

## Objective

Implement the simplest honest baseline for a binary text classification contract: a **majority-class classifier**. The model determines the most frequent class using training labels only, predicts this constant label for held-out data, and evaluates accuracy and error rate on strictly separated validation and test sets without data leakage.

> **DISCLAIMER:**  
> The fixture data provided in this project consists of six synthetic, fictional records generated solely for baseline verification and testing. These records are **NOT** real SMS messages from the SMS Spam Collection dataset or any production corpus, and must not be interpreted as evidence of real-world text classification performance.

---

## Data Schema & Contract

Each input record is a dictionary conforming to the following contract:

| Field Name | Type | Constraints | Description |
|---|---|---|---|
| `record_id` | `str` | Required, non-empty, unique string | Stable record identifier (`t1`..`t6`) |
| `text` | `str` | Required, non-empty string | Simulated message text |
| `label` | `int` | Required, binary integer: `0` or `1` | Classification target (0 = non-spam, 1 = spam) |

---

## Synthetic Fixture

The canonical fixture consists of exactly six fictional records with an equal 3 : 3 class balance:

| Record ID | Text | Label | Class Description |
|---|---|---|---|
| `t1` | `red apple` | 1 | Positive (spam) |
| `t2` | `green apple` | 1 | Positive (spam) |
| `t3` | `blue sky` | 0 | Negative (non-spam) |
| `t4` | `dark sky` | 0 | Negative (non-spam) |
| `t5` | `red fruit` | 1 | Positive (spam) |
| `t6` | `blue cloud` | 0 | Negative (non-spam) |

---

## Majority-Class Baseline Method

1. **Parameter Estimation from Training Data Only:**  
   The baseline counts label frequencies across training records only:
   $$\hat{y} = \arg\max_{c \in \{0, 1\}} \sum_{i \in \text{Train}} \mathbb{I}(y_i = c)$$
   Held-out records (validation and test) are strictly excluded from parameter estimation.

2. **Deterministic Tie-Breaking:**  
   If classes have equal counts in the training set, the tie is deterministically resolved using the `lowest` class strategy (`min(tied_classes)`), selecting class `0`.

3. **Constant Prediction:**  
   The fitted majority class is predicted for every candidate record:
   $$\hat{y}_i = \hat{y} \quad \forall i \in \text{Held-out}$$

---

## Split Strategy & Concrete Results

The canonical split partitions the six records into three disjoint sets:

| Split | Record IDs | True Labels | Count | Predicted Label | Split Accuracy |
|---|---|---|---:|---|---:|
| **Train** | `t1`, `t2`, `t3` | `1`, `1`, `0` | 3 | — (majority is 1) | — |
| **Validation** | `t4` | `0` | 1 | `1` | 0.0% (0 / 1) |
| **Test** | `t5`, `t6` | `1`, `0` | 2 | `1`, `1` | 50.0% (1 / 2) |
| **Overall Held-out** | `t4`, `t5`, `t6` | `0`, `1`, `0` | 3 | `1`, `1`, `1` | 33.3% (1 / 3) |

### Metric Definitions
- **Accuracy:** $\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Held-out Records}}$
- **Error Rate:** $\text{Error Rate} = 1.0 - \text{Accuracy} = \frac{\text{Incorrect Predictions}}{\text{Total Held-out Records}}$

Concrete results on held-out test data:
- Test Accuracy: **50.0%** (0.5000)
- Test Error Rate: **50.0%** (0.5000)
- Overall Held-out Accuracy (Val + Test): **33.33%** (1 / 3)

---

## How to Run Implementation & Tests

The pipeline is implemented using Python standard library modules (`collections.Counter`, `dataclasses`, `unittest`, `json`, `pathlib`). No external machine learning packages (e.g. scikit-learn, PyTorch, NLTK) are required.

Run the test suite from the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t003 -p "test_*.py" -v
```

---

## Scope & Limitations

- **Baseline Only:** This component implements a majority-class heuristic. It does not perform tokenization, TF-IDF vectorization, feature extraction, or probabilistic Naive Bayes modeling.
- **CPU-Only & Offline:** Operates entirely in memory on standard Python 3.
- **Synthetic Verification Data:** Uses six small simulated examples for pipeline verification. It does not establish real-world SMS spam classification benchmark metrics.
