# R41-T012: Prevent Split Leakage — Rain in Australia – logistic regression

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic deterministic formula fixture (`fixture.json`, `target = 2*x1 + 1`). These synthetic records are strictly for pipeline and split-leakage verification—they are **not** observations from the named real Rain in Australia dataset and are **not** evidence of real-world or clinical performance.

## Data Contract & Deterministic Fixture

- **Schema:** `record_id` (`str`, unique), `x1` (`int`/`float`), `x2` (`int`/`float`), `target` (`int`/`float`), plus optional `group_id`, `user_id`, and ISO `timestamp`.
- **Fixture (`fixture.json`):** 6 synthetic rows (`r1`..`r6`) with `x1 = [0, 1, 2, 3, 4, 5]`, `x2 = 1`, `target = 2*x1 + 1` (`[1, 3, 5, 7, 9, 11]`), `seed = 42`.

## Leak-Free Split Manifest

- **Train:** `["r1", "r2", "r3", "r4"]`
- **Validation:** `["r5"]`
- **Test:** `["r6"]`
- **Leakage Guards:** Rejects overlapping `record_id`s, overlapping `group_id`/`user_id` entities across splits, and non-chronological time splits.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t012 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Bounded offline CPU-only verification component only.
- Any optional real-data extension must separately document dataset source, licence, access/authentication requirements, and split design before evaluation.
