# R41-T055: Training-Only Preprocessing Contract

This bounded, standard-library-only component verifies that scaler parameters
are fitted only on the training partition and reused for later transforms.
The six fictional points are a **pipeline verification fixture**, not the
real Wholesale Customers dataset and not evidence of real-world or clinical
performance.

## Data contract

`fixture.json` is marked `"simulated": true`. Each record contains exactly
`record_id`, `x1`, and `x2`; IDs are unique strings and coordinates are finite
numbers. The points are `(0,0), (0,1), (1,0), (5,5), (5,6), (6,5)`. With fixed
centroids `(0,0)` and `(5,5)`, their assignments are `[0, 0, 0, 1, 1, 1]`.

## Split and preprocessing

The fixed-seed (`42`) split is train 4, validation 1, and test 1. The splitter
keeps the ID partitions explicit:

- Training: p4, p2, p3, p5
- Validation: p1
- Test: p6

`fit_scaler` validates that a provided split is exactly the seeded partition,
then derives means and population standard deviations from training IDs only.
The fitted state retains the seed, feature order, training IDs, means, scales,
and output dtype, and can be saved/reloaded as JSON.

The public `transform(scaler, records)` accepts only a fitted scaler and input
records; it has no fit operation or split override. It returns Python float
values with the declared `float64` dtype and an explicit `[rows, 2]` shape.
`contaminated_split.json` deliberately overlaps a held-out ID with training
and is rejected.

## Run and limits

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t055 -p "test_*.py"
```

The checks are offline, CPU-only, deterministic, and use only Python's
standard library. No real data is downloaded. A real-data extension must
separately document its source, licence, access requirements, and split design.
