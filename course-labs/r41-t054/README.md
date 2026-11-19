# R41-T054: Explicit Metric Verification Fixture

This bounded, standard-library-only component verifies binary classification
metrics and regression MAE behavior. Its six fictional text rows are a
**pipeline verification fixture**, not the real BBC News corpus and not
evidence of real-world or clinical performance.

## Data contract and split

`fixture.json` is marked `"simulated": true`. It contains exactly six records
with `record_id`, `text`, and binary `label`, using IDs t1-t6 and the specified
fictional text and labels. The fixed-seed (`42`) 4/1/1 train/validation/test
split is deterministic and disjoint:

- Training: t4, t2, t3, t5
- Validation: t1
- Test: t6

It is only a split-interface example, not model evaluation on the real BBC
dataset.

## Metric conventions

`classification_metrics` defines two explicit modes:

- `average="binary"` computes precision and recall for `positive_label` (default
  1). The required truth `[1,1,0,0]` and prediction `[1,0,1,0]` yield
  precision `0.5` and recall `0.5`.
- `average="macro"` computes each metric for labels 0 and 1, then takes their
  unweighted arithmetic mean.

For either mode, an undefined precision or recall caused by a zero denominator
is defined as `0.0`. `mean_absolute_error` is the mean absolute difference;
truth `[1,2,3]` and prediction `[1,3,2]` yield `2/3`. Shape mismatches and
unsupported averaging modes are rejected.

## Run and limits

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t054 -p "test_*.py"
```

The check is offline, CPU-only, bounded, and uses only Python's standard
library. No classifier training or real data download occurs. A real-data
extension must separately document its source, licence, access requirements,
and split design.
