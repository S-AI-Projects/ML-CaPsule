# R41-T052: Deterministic Leakage-Free Split Manifest

This bounded, standard-library-only component verifies a deterministic split
manifest for a regression fixture. The six records are a **deterministic
formula fixture** for pipeline verification, not real Sonar rocks/mines data
and not a real-world performance claim.

## Data contract and split

`fixture.json` is marked `"simulated": true`. It contains six records with
`record_id`, `x1`, `x2`, and `target`; `x1` is `[0,1,2,3,4,5]`, `x2` is always
1, and `target = 2*x1 + 1`.

The manifest records seed 42 and the exact fixed partitions:

- Training: r1-r4
- Validation: r5
- Test: r6

This prescribed split is deterministic and is not shuffled. The seed is
recorded as part of the manifest contract. The validator rejects overlapping
record IDs, incomplete or contradictory split IDs, and any group/user value
shared across partitions. If timestamps are supplied, it also requires
strictly chronological split boundaries.

## Run and limits

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t052 -p "test_*.py"
```

The check is offline, CPU-only, bounded, and uses only Python's standard
library. No real data is downloaded. A real-data extension must separately
document its source, licence, access requirements, and split design.
