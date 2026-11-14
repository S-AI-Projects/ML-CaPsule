# R41-T034: Verify the Evaluation Metric — WHO life expectancy – regression

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`). These synthetic records are strictly for pipeline and metric verification—they are **not** observations from the named real WHO life expectancy dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

- `record_id`: non-empty unique `str` (`r1`..`r6`)
- `x1`: finite scalar `float` in `{0, 1, 2, 3, 4, 5}`
- `x2`: constant feature scalar `float` (`1.0`)
- `target`: deterministic formula $2 \cdot x_1 + 1$
- `split`: split partition in `{"train", "validation", "test"}`

## Evaluation Metric Specification

### Mean Absolute Error (MAE)
$$\text{MAE}(y, \hat{y}) = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$

Canonical verification example:
- $\text{Truth} = [1.0, 2.0, 3.0]$
- $\text{Prediction} = [1.0, 3.0, 2.0]$
- $\text{MAE} = \frac{|1-1| + |2-3| + |3-2|}{3} = \frac{0 + 1 + 1}{3} = \frac{2}{3} \approx 0.6666666667$

### Zero-Denominator Handling for MAPE
When actual values $y_i = 0$, percentage error is undefined. The pipeline defines explicit strategies:
1. `epsilon`: Replace denominator with $\epsilon = 10^{-8}$ to prevent zero division.
2. `skip`: Safely omit zero-truth records from relative percentage averaging.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t034 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Bounded offline CPU-only verification component only.
- Any optional real-data extension must separately record source, licence, access requirements, and split design.
