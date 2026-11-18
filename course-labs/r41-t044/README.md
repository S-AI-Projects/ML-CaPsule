# R41-T044: Fashion-MNIST evaluation contract

This bounded contribution defines a reproducible metric contract for an image
classification pipeline and validates it using six **simulated** grayscale
images. The fixture is a CPU smoke test, not Fashion-MNIST and not evidence of
real-world or clinical performance. This contribution does not fit a random
forest or make a Fashion-MNIST performance claim.

## Data contract and split

`fixture.json` contains exactly six records. Each record has exactly these
fields:

| Field | Contract |
|---|---|
| `record_id` | Non-empty, unique string |
| `group_id` | Non-empty string identifying the split unit |
| `pixels` | 4×4 nested list of integer grayscale values in `[0, 255]` |
| `label` | Integer `0` or `1` (booleans are rejected) |

Three records are all-zero images with label `0`; the other three have a
top-left-to-bottom-right white (`255`) diagonal and label `1`. Each paired
zero/diagonal example has a declared group ID. Groups, not individual records,
are assigned explicitly to train (`group-1`), validation (`group-2`), and test
(`group-3`), preventing group overlap across splits. `schema.json` documents
the fixture envelope and row shape; `pipeline.py` enforces runtime constraints.
The tests use fixed seed `44044` only when shuffling an in-memory copy to verify
validation and split membership are independent of record order; the declared
split itself is explicit and does not rely on random assignment.

## Metric definitions

Binary precision and recall use the fixed label set `[0, 1]`. For each label,
precision is `TP / (TP + FP)` and recall is `TP / (TP + FN)`. The reported
precision and recall are unweighted macro averages over both labels, including
a label whose denominator is zero. Any per-class metric with a zero denominator
is defined as `0.0`; there is no label omission or warning-dependent behavior.
For `truth=[1, 1, 0, 0]` and `prediction=[1, 0, 1, 0]`, each class has precision
and recall `0.5`, so macro precision and recall are both `0.5`.

Regression uses mean absolute error, `sum(abs(truth - prediction)) / n`, for
non-empty equal-length finite numeric vectors. For truth `[1, 2, 3]` and
prediction `[1, 3, 2]`, MAE is `2/3`.

These are explicit metric examples, independent of a trained classifier. They
verify metric arithmetic and edge-case policy; the fixture outputs are only
pipeline verification, not Fashion-MNIST, real-world, or clinical performance.
No real-data extension is included. Any future extension must document its
source, licence, access requirements, and split design separately.

## Run checks

From the repository root, using Python 3:

```bash
python3 -m unittest discover -s course-labs/r41-t044 -p "test_*.py"
```

The implementation and tests use only the Python standard library (no NumPy,
scikit-learn, network, GPU, or dataset access). The check is offline, CPU-only,
and bounded to six records. It reports its test count and exits non-zero on a
failed assertion. Inputs, expected and actual output, the deliberate mutation
check, and tool versions are recorded in
[`work/R41-T044/`](../../work/R41-T044/).
Keep that evidence directory out of any later upstream proposal.
