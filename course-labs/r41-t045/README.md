# R41-T045: Iris k-NN preprocessing contract

This bounded component validates a numeric preprocessing contract for an
Iris-species k-NN teaching pipeline. Its eight rows are **simulated data**
invented for pipeline verification, not observations from the named Iris
dataset. It does not fit or evaluate a k-NN classifier.

## Data contract

`fixture.json` contains exactly eight records with fields in the documented
schema (`schema.json`):

| Field | Contract |
|---|---|
| `record_id` | Non-empty, unique stable row ID |
| `group_id` | Non-empty, stable group ID; one independent fixture row per group |
| `x1`, `x2` | Finite numeric values; booleans, missing values, NaN and infinity are rejected |
| `label` | Integer `0` or `1`; booleans and other labels are rejected |

The literal feature values are `x1=[0,1,2,3,4,5,6,7]` and `x2=[1,1,1,1,1,1,1,1]`;
the labels are `[0,0,0,0,1,1,1,1]`. Feature order is fixed as `["x1", "x2"]`.
Unknown or missing row fields are rejected.

## Split and preprocessing result

Seed `45` creates a deterministic, label-stratified 4/2/2 train/validation/test
split. Each split contains both labels and group IDs cannot cross boundaries:

| Split | Record IDs |
|---|---|
| Train | `iris-sim-001`, `iris-sim-004`, `iris-sim-008`, `iris-sim-006` |
| Validation | `iris-sim-002`, `iris-sim-005` |
| Test | `iris-sim-003`, `iris-sim-007` |

The scaler is fit only on the four training rows. It uses the population
standard deviation and replaces a zero standard deviation with `1` so a
constant feature maps to zero. The training means are `[3.75, 1.0]`, scales are
approximately `[2.5860201082, 1.0]`, and validation transforms to
`[[-1.06341013795, 0.0], [0.09667364890, 0.0]]`. The test transform is
`[[-0.67671554233, 0.0], [0.87006284014, 0.0]]`. `transform` only applies stored fit parameters;
the public transform operation accepts no fit data or fit mode. Split
validation rejects repeated record assignments and group contamination before
the pipeline fits.

## Run checks

From the repository root, using Python 3:

```bash
python3 -m unittest discover -s course-labs/r41-t045 -p "test_*.py"
```

The implementation uses only the Python standard library: no NumPy,
scikit-learn, network access, real dataset, GPU, or paid service is required.
Transformed output uses Python binary64 `float` values and declares its
`float64` dtype and two-dimensional shape in the result. The check is bounded
to eight rows and exits non-zero on assertion failure. Inputs, expected and
actual output, and tool versions are recorded under `work/R41-T045/`.

These results establish pipeline behavior only, not Iris-species accuracy,
real-world generalization, or clinical performance. No real-data extension is
included. A future extension must document its source, licence, access
requirements, and leakage-aware split design separately before making
dataset-specific claims.
