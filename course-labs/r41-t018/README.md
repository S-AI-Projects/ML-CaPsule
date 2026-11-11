# R41-T018: Explain an Error Case: Titanic Survival – Decision Tree

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic classification fixture (`fixture.json`). These records verify structured error analysis reporting, failure reason categorization, and testable misclassification injection—they are **not** observations from the real Titanic survival dataset and are **not** evidence of real-world or clinical performance.

## Data Contract & Synthetic Fixture

- **Schema:** `(record_id, group_id, x1, x2, label)`
- **Fixture:** 8 synthetic rows ($x1 \in [0..7]$, $x2 = 1.0$, label $\in \{0, 1\}$).
- **Fixed Stratified Split:**
  - **Train:** `["r1", "r2", "r5", "r6"]`
  - **Validation:** `["r3", "r7"]`
  - **Test:** `["r4", "r8"]`

## Error Analysis Report Schema

The report records:
1. **`record_id`:** Stable identifier for the misclassified row.
2. **`expected`:** Ground truth label.
3. **`predicted`:** Classifier output label.
4. **`reason_category`:** Root-cause categorization (`BOUNDARY_AMBIGUITY`, `MISCLASSIFICATION_NOISE`, or `CORRECT_PREDICTION`).
5. **`feature_values`:** Captured feature state at prediction time.

## Deliberate Failure Injection & Verification

- To make the reporting logic explicitly testable, an error is injected on test sample `r4` (injected prediction `1`, expected `0`).
- Unit tests verify that `r4` appears in the error report, correct sample `r8` is omitted, and aggregate counts match row details.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t018 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Offline CPU-only error case explanation verification component only.
- Any optional real-data extension must separately record source, license, access requirements, and decision tree splitting criteria.
