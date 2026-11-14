# R41-T033: Compare a Minimal Baseline — Palmer penguins – logistic regression

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`). These synthetic records are strictly for pipeline verification and baseline comparison—they are **not** observations from the named real Palmer penguins dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

- `record_id`: non-empty unique `str` (`r1`..`r6`)
- `x1`: finite scalar `float` in `{0, 1, 2, 3, 4, 5}`
- `x2`: constant feature scalar `float` (`1.0`)
- `target`: deterministic formula $2 \cdot x_1 + 1$
- `split`: split partition in `{"train", "validation", "test"}`

## Deterministic Formula Fixture

- $x_1 = [0, 1, 2, 3, 4, 5]$, $x_2 = 1.0$, $\text{target} = 2 \cdot x_1 + 1$
- Training set: `r1`..`r4` with targets $[1, 3, 5, 7]$
- Validation set: `r5` with target $9.0$
- Test set: `r6` with target $11.0$

## Minimal Baseline Definition

The simplest honest regression baseline fits solely on the training rows:
$$\bar{y}_{\text{train}} = \frac{1+3+5+7}{4} = 4.0$$
Predictions for any held-out input are $\hat{y} = 4.0$.

Evaluation on held-out splits:
- Validation MAE on `r5`: $|9.0 - 4.0| = 5.0$
- Test MAE on `r6`: $|11.0 - 4.0| = 7.0$

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t033 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Bounded offline CPU-only verification component only.
- Any optional real-data extension must separately record source, licence, access requirements, and split design.
