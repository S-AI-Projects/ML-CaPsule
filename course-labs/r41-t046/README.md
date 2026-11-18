# R41-T046: Taiwan credit-default classifier smoke test

This bounded contribution exercises a classification train/inference contract
for the Taiwan credit-default gradient-boosting project area. Its eight rows
are **simulated data** invented for pipeline verification, not observations
from the Taiwan credit-default dataset. It uses an intentionally naive
training-majority constant baseline, not gradient boosting, and does not
measure predictive usefulness.

## Data and split contract

`fixture.json` contains exactly eight records matching `schema.json`:

| Field | Contract |
|---|---|
| `record_id` | Non-empty, unique stable row ID |
| `group_id` | Non-empty, stable group ID; one independent row per group |
| `x1`, `x2` | Finite numeric inputs; booleans, missing values, NaN and infinity are rejected |
| `label` | Integer `0` or `1`; booleans and other values are rejected |

The literal values are `x1=[0,1,2,3,4,5,6,7]`, `x2=[1,1,1,1,1,1,1,1]`,
and `label=[0,0,0,0,1,1,1,1]`. Feature order is `["x1", "x2"]`; unknown or
missing fields are rejected.

Seed `46` makes a reproducible label-stratified 4/2/2 train/validation/test
split:

| Split | Record IDs |
|---|---|
| Train | `credit-sim-003`, `credit-sim-004`, `credit-sim-008`, `credit-sim-005` |
| Validation | `credit-sim-002`, `credit-sim-007` |
| Test | `credit-sim-001`, `credit-sim-006` |

Every split contains both labels, each group remains in one split, and every
record is assigned exactly once. Evaluation explicitly rejects record overlap
with training IDs.

## Smoke-test result and limits

The model predicts the most common training label (ties choose `0`) and returns
the training positive-label fraction as a constant probability. Training is
limited to four rows and uses zero parameter-update steps; the recorded step
count is `0`, within the one-epoch/20-step ceiling. For this balanced training
fixture, predictions are `[0,0]` and probabilities `[0.5,0.5]` for each
held-out split. Binary cross-entropy is finite (`0.6931471805599453`) for both
validation and test. Outputs expose one-dimensional prediction and probability
shapes, and the test asserts finite values and loss.

The check is offline, CPU-only, bounded to eight fixture rows, and uses only the
Python standard library. Run from the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t046 -p "test_*.py"
```

Inputs, expected and actual output, and tool versions are recorded under
`work/R41-T046/`. These are pipeline smoke-test results only, not Taiwan credit
default performance or evidence of real-world or clinical usefulness. No
real-data extension is included. A future extension must document its data
source, licence, access requirements, and leakage-aware split design
separately before making dataset-specific claims.
