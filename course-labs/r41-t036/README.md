# R41-T036: Add a Bounded CPU Smoke Run — FIFA players – rating regression

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`). These synthetic records are strictly for pipeline and interface correctness smoke testing—they are **not** observations from the named real FIFA players dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

- `record_id`: non-empty unique `str` (`r1`..`r6`)
- `x1`: finite scalar `float` in `{0, 1, 2, 3, 4, 5}`
- `x2`: constant feature scalar `float` (`1.0`)
- `target`: deterministic formula $2 \cdot x_1 + 1$
- `split`: split partition in `{"train", "validation", "test"}`

## Bounded CPU Smoke Run Invariants

1. **Hardware & Execution Constraints:** Zero GPU dependencies, zero external credentials, zero network access. Pure offline CPU execution.
2. **Sample Bound:** Strictly $\le 32$ samples (fixture contains 4 train records, 1 validation record, 1 test record).
3. **Epoch & Step Bound:** Fixed seed (`42`), $\le 1$ training epoch, and $\le 20$ update steps (configured to 15 gradient steps).
4. **Finite Outputs:** Loss and predictions are guaranteed to evaluate to finite numerical values (no `NaN`, no `Inf`).
5. **Model Classification:** Explicitly identified as a bounded deterministic CPU baseline.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t036 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Bounded offline CPU-only verification component only.
- Any optional real-data extension must separately record source, licence, access requirements, and split design.
