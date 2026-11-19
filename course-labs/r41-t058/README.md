# R41-T058: Clustering Error-Analysis Fixture

This bounded, standard-library-only component verifies an error-report
interface for an unsupervised clustering example. All six points and the
injected prediction are synthetic; results are pipeline verification only,
not evidence of real-world or clinical performance on an online-retail dataset.

## Data contract

`fixture.json` is explicitly marked `"simulated": true`. Its records contain
exactly `record_id`, `x1`, and `x2`; IDs are unique strings and coordinates are
finite numbers. The six points, in fixture order, are:

`(0,0), (0,1), (1,0), (5,5), (5,6), (6,5)`.

The fixed centroids are cluster 0 at `(0,0)` and cluster 1 at `(5,5)`.
Nearest-centroid assignments in fixture order are `[0, 0, 0, 1, 1, 1]`.
Equal-distance ties, if encountered, go to the lower centroid ID.

## Error report

`injected_predictions.json` deliberately assigns point `p6` to cluster 0,
although its expected nearest-centroid assignment is cluster 1.
`error_analysis_report.json` is the generated, test-checked report: it contains
only the incorrect row, including stable ID, expected value, predicted value,
and reason category. Its aggregate error count equals the number of detailed
rows, and correct predictions are omitted.

The fixed-seed (`42`) 4/1/1 train/validation/test partition is explicit and
disjoint:

- Training: p4, p2, p3, p5
- Validation: p1
- Test: p6

Centroids are provided constants rather than fitted from any partition; this
split is recorded to make the tiny fixture's evaluation boundaries explicit,
not to claim model training or generalization.

## Run and limits

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t058 -p "test_*.py"
```

The checks are offline, CPU-only, deterministic, and use only Python's standard
library. No real online-retail data is downloaded or used. A real-data
extension must document its source, licence, access requirements, and split
design separately.
