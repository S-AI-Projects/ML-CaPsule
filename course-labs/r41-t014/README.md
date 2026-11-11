# R41-T014: Verify the Evaluation Metric — Adult income (UCI) – gradient boosting

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic classification fixture (`fixture.json`). These records are strictly for evaluation metric and stratified-split verification—they are **not** observations from the real UCI Adult income dataset and are **not** evidence of real-world or clinical performance.

## Data Contract & Synthetic Fixture

- **Schema:** `(record_id, group_id, x1, x2, label)`
- **Fixture:** 8 synthetic records (`r1`..`r8`) with x1 = [0, 1, 2, 3, 4, 5, 6, 7], x2 = 1, and label = [0, 0, 0, 0, 1, 1, 1, 1].
- **Fixed Stratified Split:**
  - **Train:** ["r1", "r2", "r5", "r6"] (two 0s, two 1s)
  - **Validation:** ["r3", "r7"] (one 0, one 1)
  - **Test:** ["r4", "r8"] (one 0, one 1)

## Metric Cross-Check & Zero-Denominator Policy

- **Benchmark Check:** truth = [1, 1, 0, 0], prediction = [1, 0, 1, 0]
  - TP = 1, FP = 1, FN = 1, TN = 1
  - Accuracy = 0.5
  - Precision = 0.5
  - Recall = 0.5
  - F1 = 0.5
- **Zero-Denominator Rule:** When denominator is zero, default output is bounded to 0.0 (configurable via zero_division).

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t014 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Offline CPU-only bounded verification component only.
- Any optional real-data extension must separately record source, license, access requirements, and stratified split design.
