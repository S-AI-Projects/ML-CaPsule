# R41-T056: Tiny Classification Pipeline Smoke Test

This bounded, standard-library-only decision-stump smoke test verifies the
classification interface, output shape, finite losses, and step accounting.
Its eight rows are synthetic pipeline-verification data, not observations
from the real Car Evaluation dataset and not evidence of useful real-world or
clinical performance.

## Data contract

`fixture.json` is marked `"simulated": true`. Each row contains exactly
`record_id`, `group_id`, `x1`, `x2`, and `label`. Row IDs are unique strings;
each group contains one example of each binary label.

| record IDs | x1 | x2 | label | group |
|---|---:|---:|---:|---|
| r0-r3 | 0-3 | 1 | 0 | g0-g3 |
| r4-r7 | 4-7 | 1 | 1 | g0-g3 |

## Split and smoke model

Seed `42` shuffles group IDs only; records remain in fixture order and group
members never cross partitions. The stratified split is 4/2/2:

- Training: r1, r2, r5, r6 (groups g1, g2)
- Validation: r3, r7 (group g3)
- Test: r0, r4 (group g0)

The CPU-only model is a deterministic one-level decision stump. It evaluates
at most 20 candidate thresholds in one pass; the fixture evaluates 3. This is
not an iterative training loop, and the reported training/evaluation losses
only prove finite interface behavior, not model quality.

## Run and limits

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t056 -p "test_*.py"
```

The check is offline, bounded to eight fixture rows, and uses only Python's
standard library. No GPU, credentials, or data download is required. Any
real-data extension must separately document its source, licence, access
requirements, and split design.
