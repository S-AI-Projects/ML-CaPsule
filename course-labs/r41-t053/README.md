# R41-T053: Training-Mean Regression Baseline

This bounded, standard-library-only component verifies an honest baseline for
a regression interface. Its six rows are a **deterministic formula fixture**
for pipeline verification, not real laptop-price observations and not a
real-world performance claim.

## Data contract and split

`fixture.json` is marked `"simulated": true`. It contains exactly six rows
with `record_id`, `x1`, `x2`, and `target`; `x1` is `[0,1,2,3,4,5]`, `x2` is
always 1, and `target = 2*x1 + 1`.

The fixed partition is:

- Training: r1-r4
- Validation: r5
- Test: r6

Seed 42 is recorded in the split/baseline contract. This specified split is
fixed directly and never shuffles the rows.

## Baseline and evaluation

The baseline parameter is the mean training target:
`(1 + 3 + 5 + 7) / 4 = 4`. The fitted baseline uses only r1-r4 and predicts 4
for both held-out rows. MAE is reported separately for validation (5), test
(7), and combined held-out data (6). No real dataset is downloaded or used.

## Run and limits

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t053 -p "test_*.py"
```

The check is offline, CPU-only, bounded, and uses only Python's standard
library. Any real-data extension must separately document its source, licence,
access requirements, and split design.
