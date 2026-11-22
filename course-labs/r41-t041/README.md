# R41-T041: Concrete-strength regression input contract

This is a bounded dataset-contract contribution in the repository's
`course-labs/` namespace. It defines and validates one row with exactly
`record_id`, `x1`, `x2`, and `target`; it is not a reproduction of a full
concrete-strength project.

## Data fixture and limits

`fixture.json` is a **simulated deterministic formula fixture**, not a real
concrete-strength dataset. It has six literal rows with `x1 = [0, 1, 2, 3, 4,
5]`, `x2 = 1`, and `target = 2 * x1 + 1`. IDs `r1`–`r4` are training rows, `r5`
is validation, and `r6` is test. Split membership is stored separately from
each row so that the row schema stays exactly the four declared fields.

The fixture verifies schema and split plumbing only. It is not evidence of
real-world concrete-strength performance, model accuracy, or fitness for
engineering decisions. No real-data extension is included. Any future
extension must separately document the dataset source, its licence, access
requirements, and a leakage-aware split design before reporting results.

## Contract and validation behavior

`schema.json` declares the required fields and rejects additional properties.
`pipeline.py` enforces runtime conditions that JSON Schema alone does not cover:

- `record_id` must be a non-empty string and unique within the input.
- `x1`, `x2`, and `target` must be finite integer or floating-point scalars.
  Booleans, numeric text, arrays/tensors, nulls, NaN, and infinities are invalid.
- Missing fields, unknown fields, malformed rows, and duplicate IDs produce
  issues containing the record ID, field, and a specific reason.
- Validation returns a report whose row count includes invalid rows. It never
  coerces values or silently filters records.

This component uses only the Python standard library. There is no model fit, no
network or dataset access, and no dependency on NumPy or scikit-learn.

## Run checks

From the repository root:

```sh
python3 -m unittest discover -s course-labs/r41-t041 -p "test_*.py"
```

The tests are offline and CPU-only. They use a fixed seed only to create a
repeatable shuffled copy when checking input-order independence; the fixture
split itself is explicit and never randomized. An assertion failure makes
`unittest` exit non-zero.

Run evidence for this check is stored in [`work/R41-T041/`](../../work/R41-T041/)
for the coursework branch. Do not carry that evidence directory into any later
upstream proposal.