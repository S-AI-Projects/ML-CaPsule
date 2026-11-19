# R41-T022: Prevent Split Leakage — Abalone age – regression

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`). These synthetic records are strictly for pipeline and split-manifest reproducibility verification—they are **not** observations from the named real Abalone dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

The dataset contract defines the required regression fields for every sample record:

| Field | Type | Constraint | Description |
|---|---|---|---|
| `record_id` | `str` | Non-empty, unique | Stable record identifier (`r1`..`r6`) |
| `group_id` | `str` | Non-empty | Cohort or individual identifier for group isolation |
| `timestamp` | `int` | Monotonic | Chronological timestamp for temporal ordering |
| `x1` | `float` | $[0, 5]$ | Primary regression feature |
| `x2` | `float` | $1.0$ | Constant secondary feature |
| `target` | `float` | $2 \cdot x_1 + 1$ | Deterministic continuous regression target |
| `split` | `str` | `train` \| `validation` \| `test` | Manifest split assignment |

## Synthetic Formula Fixture

The fixture (`course-labs/r41-t022/fixture.json`) defines six synthetic rows strictly generated via $target = 2 \cdot x_1 + 1$:

| `record_id` | `group_id` | `timestamp` | `x1` | `x2` | `target` | Partition |
|---|---|---|---|---|---|---|
| `r1` | `grp_1` | 1001 | 0.0 | 1.0 | 1.0 | `train` |
| `r2` | `grp_1` | 1002 | 1.0 | 1.0 | 3.0 | `train` |
| `r3` | `grp_2` | 1003 | 2.0 | 1.0 | 5.0 | `train` |
| `r4` | `grp_2` | 1004 | 3.0 | 1.0 | 7.0 | `train` |
| `r5` | `grp_3` | 1005 | 4.0 | 1.0 | 9.0 | `validation` |
| `r6` | `grp_4` | 1006 | 5.0 | 1.0 | 11.0 | `test` |

## Deterministic Split Manifest Contract

The split manifest (`create_deterministic_split` and `validate_manifest` in `pipeline.py`) enforces strict leakage-free partitioning:
1. **Disjointness:** $\text{train} \cap \text{validation} = \emptyset$, $\text{train} \cap \text{test} = \emptyset$, $\text{validation} \cap \text{test} = \emptyset$. Intentionally overlapping IDs are rejected.
2. **Complete Coverage:** $\text{train} \cup \text{validation} \cup \text{test} = \{\text{r1}, \text{r2}, \text{r3}, \text{r4}, \text{r5}, \text{r6}\}$.
3. **Group Isolation:** Records sharing a `group_id` (`grp_1`, `grp_2`) reside entirely within one split; cross-split group leakage is rejected.
4. **Chronological Ordering:** Monotonic time guarantees $\max(t_{\text{train}}) \le \min(t_{\text{val}}) \le \min(t_{\text{test}})$, preventing future-to-past lookahead leakage.
5. **Exact Manifest Record IDs:**
   - `train`: `["r1", "r2", "r3", "r4"]`
   - `validation`: `["r5"]`
   - `test`: `["r6"]`

## Regression Model Baseline

The bundled CPU-only `SimpleLinearRegression` fits closed-form ordinary least squares on the training partition:
- Recovered parameters: $\text{slope} = 2.0$, $\text{intercept} = 1.0$.
- Validation evaluation on `r5` ($x_1=4.0$): predicted $9.0$, actual $9.0$, $\text{MSE} = 0.0$.
- Test evaluation on `r6` ($x_1=5.0$): predicted $11.0$, actual $11.0$, $\text{MSE} = 0.0$.

## Automated Verification

```bash
python3 -m unittest discover -s course-labs/r41-t022 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- **Limits:** This is an offline, CPU smoke fixture designed strictly for pipeline verification and split manifest assertions. It does not reflect complex biological abalone measurements.
- **Optional Real-Data Extension:** Any transition to the full UCI Abalone dataset must separately document:
  - **Source:** UCI Machine Learning Repository (Abalone dataset, donor: Warwick J. Nash, 1995).
  - **Licence:** Creative Commons Attribution 4.0 International (CC BY 4.0).
  - **Access Requirements:** Unrestricted public download via UCI ML Repository HTTPS endpoint.
  - **Split Design:** Explicit demographic/group-aware train/validation/test split ensuring non-overlapping rings/age cohorts where applicable.
