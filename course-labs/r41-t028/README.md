# R41-T028 · Twitter Airline Sentiment – Naive Bayes: Error-Analysis Report

**Task:** R41-T028 · 8 of 10 | **Student:** Sharon Robinson | **Branch:** `R41-T028` | **Assigned working branch:** `student/e26b07f0819`  
**Repository:** [S-AI-Projects/ML-CaPsule](https://github.com/S-AI-Projects/ML-CaPsule)

---

## Goal

Create and verify a small, structured **error-analysis report** for the **Twitter airline sentiment – Naive Bayes** teaching component (`course-labs/r41-t028`):

- Emit stable `record_id`, `expected_value`, `predicted_value`, `split`, and controlled `reason_category` for misclassified records while excluding correct predictions.
- Inject **one deliberately wrong prediction** (`t5`: expected `1`, flipped to `0`) so the error-analysis pipeline is end-to-end testable with nonzero error rows.
- Enforce exact reconciliation between aggregate error counters (`error_count`, `correct_count`, `false_positive_count`, `false_negative_count`, `counts_by_reason`, `counts_by_split`) and the detailed `error_rows`.

> **SYNTHETIC DATA DISCLAIMER:** All six records (`t1..t6`) in `fixture.json` are a **fictional synthetic text pipeline fixture**. They are **not** a real Twitter airline sentiment corpus, and all results are strictly **pipeline verification**, not evidence of real-world or clinical performance.

---

## Data Contract

Defined in [`schema.json`](schema.json) and instantiated literally in [`fixture.json`](fixture.json).

| record_id | text | label | split | Notes |
|-----------|------|-------|-------|-------|
| `t1` | `"red apple"` | `1` | `train` | Training positive phrase |
| `t2` | `"green apple"` | `1` | `train` | Training positive phrase |
| `t3` | `"blue sky"` | `0` | `train` | Training negative phrase |
| `t4` | `"dark sky"` | `0` | `train` | Training negative phrase |
| `t5` | `"red fruit"` | `1` | `validation` | Held-out (`"fruit"` OOV); target for deliberate error injection |
| `t6` | `"blue cloud"` | `0` | `test` | Held-out (`"cloud"` OOV); remains correctly predicted (`0`) |

---

## Pipeline Verification & Error-Analysis Results (Synthetic Fixture Only)

1. **Multinomial Naive Bayes baseline (`alpha=1.0`, `seed=42`)**:
   - Fitted strictly on `train` (`t1..t4`), vocabulary = `{"apple", "blue", "dark", "green", "red", "sky"}` (held-out tokens `"fruit"` and `"cloud"` never enter the training vocabulary).
   - Clean predictions: `t1=1, t2=1, t3=0, t4=0, t5=1, t6=0` (0 errors on the clean held-out split).
2. **Testable error-analysis report with one deliberately wrong prediction (`t5: 1 -> 0`)**:

| record_id | split | text | expected | predicted | reason_category | oov_tokens |
|-----------|-------|------|----------|-----------|-----------------|------------|
| `t5` | `validation` | `red fruit` | `1` | `0` | `injected_test_error` | `["fruit"]` |

- **Held-out (`validation` + `test`) aggregates**: `total_evaluated = 2`, `correct_count = 1` (`t6` excluded from error rows), `error_count = 1`, `false_positive_count = 0`, `false_negative_count = 1`, `counts_by_reason = {"injected_test_error": 1}`, `counts_by_split = {"validation": 1}`.
- **Full fixture (`t1..t6`) aggregates**: `total_evaluated = 6`, `correct_count = 5` (`t1, t2, t3, t4, t6` excluded), `error_count = 1` (`t5` only).

---

## Running the Automated Check

Uses only the Python 3 standard library (offline, CPU-only, zero external dependencies):

```bash
python3 -m unittest discover -s course-labs/r41-t028 -p "test_*.py" -v
```

### Unit Test Coverage (17 tests)

- `TestSyntheticTextFixtureContract` (3): synthetic disclaimer, literal 6 records `t1..t6`, schema/reason alignment
- `TestExplicitTrainValTestAndNaiveBayes` (3): disjoint splits, training-only vocabulary, exact clean predictions
- `TestErrorAnalysisReportAndInjectedError` (5): 0-error clean report, injected `t5` appears while correct `t6` does not, full-fixture exclusion of all 5 correct records, exact aggregate-to-row reconciliation, natural reason categories (`oov_token_shift`, `false_positive_lexical`, `false_negative_lexical`) & Markdown output
- `TestFailureCasesAndGuards` (5): non-train record in `fit`, overlapping `record_id`, missing/extra prediction IDs, invalid labels/empty text/unfitted predict, `ErrorRow` rejecting correct predictions
- `TestDeliberateMutationGuard` (1): proves altered expected error counts fail assertions

---

## Limits

- Uses 6 two-word synthetic phrases (`t1..t6`) purely to verify Naive Bayes split separation and error-report accounting; carries zero linguistic complexity (negation, sarcasm, URLs, hashtags, class imbalance) of real airline tweets.
- Controlled reason categories (`injected_test_error`, `oov_token_shift`, `false_positive_lexical`, `false_negative_lexical`) are deterministic heuristics for pipeline verification, not manual linguistic error adjudication.

---

## Optional Real-Data Extension Contract

Any future extension on real airline sentiment tweets **must separately document** its provenance and distinguish real-corpus metrics from synthetic fixture results:

| Requirement | Specification |
|-------------|---------------|
| **Source** | Twitter US Airline Sentiment dataset (Crowdflower / Figure Eight, 2015) |
| **Licence** | CC BY-NC-SA 4.0 (verify exact redistribution terms on Kaggle/source before use) |
| **Access requirements** | Download via Kaggle (`crowdflower/twitter-airline-sentiment`, requires user account/API token; keep raw CSVs out of Git per `.gitignore`) |
| **Split design** | ~14,640 tweets across 3 sentiment classes (`negative`, `neutral`, `positive`); use chronological or user-grouped stratified train/validation/test splits to avoid author and retweet leakage across splits |

---

## Tool Versions

- **Runtime requirement**: Python 3 standard library only (verified offline on Python `3.15.2`, macOS arm64)
