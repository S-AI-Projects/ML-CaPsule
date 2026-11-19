# R41-T051: Regression Dataset Contract

This bounded, standard-library-only component validates a regression input
contract. The six records are a **deterministic formula fixture** for pipeline
verification, not real CarDekho used-car observations and not a real-world
performance claim.

## Data contract

`fixture.json` is marked `"simulated": true`. Each row contains exactly
`record_id`, `x1`, `x2`, and `target`; IDs are unique strings and numeric
fields must be finite scalar numbers (vectors/tensors are unsupported). The
fixture uses `x1=[0,1,2,3,4,5]`, `x2=1`, and `target=2*x1+1`.

The validator accumulates errors rather than dropping invalid rows. Each
diagnostic includes the available record ID, row index, field, and reason.
Missing IDs have `record_id: null` and remain locatable by row index.

## Split manifest

Seed 42 is recorded in the deterministic manifest:

- Training: r1-r4
- Validation: r5
- Test: r6

This specified split is fixed and is not shuffled. The manifest checks exact
coverage and rejects overlapping IDs.

## Run and limits

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t051 -p "test_*.py"
```

The check is offline, CPU-only, bounded, and uses only Python's standard
library. No real data is downloaded. A real-data extension must separately
document its source, licence, access requirements, and split design.
