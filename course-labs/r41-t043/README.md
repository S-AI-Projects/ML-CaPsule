# R41-T043: majority-class text baseline

This bounded contribution adds the simplest classification baseline for the
language-detection project area. The six records are **fictional simulated
data**, used only for pipeline verification. The binary labels are arbitrary
and are not language identifiers.

## Data contract

`fixture.json` contains exactly six records with this schema:

| Field | Contract |
|---|---|
| `record_id` | Non-empty, unique string |
| `text` | Non-empty string |
| `label` | Integer `0` or `1` (boolean values are rejected) |

`schema.json` describes each record. The literal fixture is `t1`–`t6` with
texts `red apple`, `green apple`, `blue sky`, `dark sky`, `red fruit`, and
`blue cloud`; the corresponding labels are `1, 1, 0, 0, 1, 0`.

## Baseline and result

With seed `43`, the seeded 4/1/1 split is:

| Split | Record IDs |
|---|---|
| Train | `t6`, `t5`, `t4`, `t2` |
| Validation | `t3` |
| Test | `t1` |

Only the training labels fit the baseline. Their counts tie (two per label),
so the deterministic tie rule chooses label `0`. The validation prediction is
`[0]` with accuracy `1.0`; the held-out test prediction is `[0]` for true
label `[1]`, with accuracy `0.0`. Accuracy is the fraction of correct
predictions on the named split.

These are synthetic pipeline-verification results, not language-detection
accuracy, a real-corpus result, or evidence of real-world or clinical
performance. This baseline does not inspect the text, extract character
n-grams, or demonstrate that language detection works; it is only a comparison
floor for a later model.

## Run checks

From the repository root, with Python 3:

```bash
python3 -m unittest discover -s course-labs/r41-t043 -p "test_*.py"
```

The implementation and tests use only the Python standard library (no
third-party or optional dependency versions). The check is offline, CPU-only,
and bounded to six fixture rows. A failed assertion makes `unittest` exit
non-zero. Test inputs, expected and actual output, and tool versions are kept
in `work/R41-T043/`.

No real-data extension is included. Any future extension must separately
document its data source, licence, access requirements, and split design
before reporting corpus-specific performance.
