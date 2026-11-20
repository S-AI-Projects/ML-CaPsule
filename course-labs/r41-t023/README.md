# R41-T023: Compare a Minimal Baseline — Auto MPG – regression

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`). These synthetic records are strictly for pipeline and baseline comparison verification—they are **not** observations from the named real Auto MPG dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

The dataset contract defines the required continuous regression fields for every sample record:

| Field | Type | Constraint | Description |
|---|---|---|---|
| `record_id` | `str` | Non-empty, unique | Stable record identifier (`r1`..`r6`) |
| `x1` | `float` | $[0, 5]$ | Primary regression feature variable |
| `x2` | `float` | $1.0$ | Constant secondary feature variable |
| `target` | `float` | $2 \cdot x_1 + 1$ | Deterministic continuous regression target |
| `split` | `str` | `train` \| `validation` \| `test` | Partition assignment |

## Synthetic Formula Fixture

The fixture (`course-labs/r41-t023/fixture.json`) contains six synthetic rows strictly generated via $target = 2 \cdot x_1 + 1$:

| `record_id` | `x1` | `x2` | `target` | Split Partition |
|---|---|---|---|---|
| `r1` | 0.0 | 1.0 | 1.0 | `train` |
| `r2` | 1.0 | 1.0 | 3.0 | `train` |
| `r3` | 2.0 | 1.0 | 5.0 | `train` |
| `r4` | 3.0 | 1.0 | 7.0 | `train` |
| `r5` | 4.0 | 1.0 | 9.0 | `validation` |
| `r6` | 5.0 | 1.0 | 11.0 | `test` |

## Simplest Honest Baseline (Training-Mean)

The simplest honest regression baseline (`TrainingMeanBaseline` in `pipeline.py`) computes the scalar target mean exclusively from the training partition:
$$\bar{y}_{\text{train}} = \frac{1.0 + 3.0 + 5.0 + 7.0}{4} = 4.0$$

### Parameter Isolation Guarantee
- Training IDs `r1..r4` are the sole records used to estimate $\bar{y}_{\text{train}}$.
- Zero held-out data (validation `r5`, test `r6`) is observed during parameter estimation.

### Held-out Evaluation Results

| Split | Record IDs | Actual Target | Baseline Prediction | MAE | MSE | RMSE |
|---|---|---|---|---|---|---|
| `validation` | `r5` | 9.0 | 4.0 | 5.0 | 25.0 | 5.0 |
| `test` | `r6` | 11.0 | 4.0 | 7.0 | 49.0 | 7.0 |

### Oracle Contrast (Feature-Based Linear Regression)
A closed-form least-squares linear model (`LinearRegressionComparator`) fitting $y = 2 \cdot x_1 + 1$ achieves $\text{MSE} = 0.0$ on the same held-out partitions, demonstrating the performance gap between naive constant prediction and true signal recovery.

## Automated Verification

```bash
python3 -m unittest discover -s course-labs/r41-t023 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- **Limits:** This is an offline, CPU smoke fixture designed strictly for pipeline verification and baseline comparison assertions. It does not reflect real automobile fuel efficiency dynamics.
- **Optional Real-Data Extension:** Any transition to the full UCI Auto MPG dataset must separately document:
  - **Source:** StatLib / CMU / UCI Machine Learning Repository (Auto MPG dataset, 1993).
  - **Licence:** Public Domain / CC BY 4.0.
  - **Access Requirements:** Unrestricted public download via UCI ML Repository HTTPS endpoint.
  - **Split Design:** Explicit chronological or origin-stratified 80/10/10 train/validation/test split with strict multi-collinearity checks.
