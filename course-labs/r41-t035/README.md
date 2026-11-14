# R41-T035: Make Preprocessing Explicit — World happiness report – EDA + regression

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`). These synthetic records are strictly for pipeline and preprocessing contract verification—they are **not** observations from the named real World happiness report dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

- `record_id`: non-empty unique `str` (`r1`..`r6`)
- `x1`: finite scalar `float` in `{0, 1, 2, 3, 4, 5}`
- `x2`: constant feature scalar `float` (`1.0`)
- `target`: deterministic formula $2 \cdot x_1 + 1$
- `split`: split partition in `{"train", "validation", "test"}`

## Explicit Preprocessing Contract

1. **Training-Only Fitting:** Missing-value imputation and feature scaling parameters ($\mu, \sigma$) are computed exclusively over the training partition (`r1`..`r4`).
   - $\mu_{x1} = 1.5$, $\sigma_{x1} = \sqrt{1.25} \approx 1.11803$
   - $\mu_{x2} = 1.0$, $\sigma_{x2} = 1.0$ (constant unit fallback)
2. **Leakage Prevention:** Transforming held-out rows (`validation`, `test`) uses pre-fitted training statistics without mutating or updating internal parameters.
3. **State Invariants:** Calling `transform()` before `fit()` is blocked with a `RuntimeError`.
4. **Contamination Rejection:** Malformed non-numeric types or non-finite tokens (`NaN`, strings) are rejected with a `ValueError`.
5. **Output Invariant:** The preprocessor guarantees strict feature ordering (`["x1", "x2"]`), shape preservation `(N, 2)`, and `float64` output data type.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t035 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Bounded offline CPU-only verification component only.
- Any optional real-data extension must separately record source, licence, access requirements, and split design.
