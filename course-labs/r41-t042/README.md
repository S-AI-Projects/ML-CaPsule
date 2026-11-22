# R41-T042: deterministic text split manifest

This bounded contribution provides a small deterministic split-manifest utility for a sentiment-analysis pipeline. The fixture contains six **fictional, synthetic** records (`t1`–`t6`) and is used only for pipeline verification. It is not the IMDB review corpus and no model is trained here.

## Contract and result

Each input record requires:

| Field | Contract |
|---|---|
| `record_id` | Non-empty string, unique across the full input |
| `text` | Non-empty string |
| `label` | Integer `0` or `1` (`bool` is not accepted) |

Optional `group_id` and `user_id` fields are treated as group boundaries. If either is present, it must be present on every record; connected records sharing either key remain in one split. Pass `time_field` explicitly (for example `timestamp`) to use chronological, contiguous splits. The time field must exist on every row and contain ISO-8601 datetime strings. If group isolation and chronology cannot both be maintained, the utility rejects the input rather than leaking or silently dropping records.

With seed `41`, the checked-in manifest is:

| Split | Record IDs |
|---|---|
| Train | `t5`, `t6`, `t1`, `t2` |
| Validation | `t3` |
| Test | `t4` |

Without time metadata, independent group components are shuffled using the fixed seed. Split sizes target approximately 67%/17%/16%; indivisible groups can make exact ratios impossible. The manifest includes every input ID exactly once. `validate_manifest` rejects duplicate/unknown/missing IDs, groups spanning splits, and non-chronological manifests.

## Run checks

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t042 -p "test_*.py"
```

The component uses only the Python standard library; it requires no network, external data, GPU, or third-party packages. Evidence and observed tool versions are recorded under `work/R41-T042/` for the coursework check.

## Limits and data provenance

The six rows are invented example text, not downloaded or derived from user reviews. The seeded split demonstrates deterministic ID assignment, not a fitted logistic-regression result, sentiment quality, real-world generalization, or clinical performance. No real-data extension is included. Any future IMDB extension must separately document its source, licence, access requirements, and leakage-aware split design before making corpus-specific claims.