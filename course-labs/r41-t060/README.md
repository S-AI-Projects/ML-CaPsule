# R41-T060: 20 Newsgroups Topic-Clustering Pipeline Fixture

This is a bounded saved-model-interface component for the
`20 Newsgroups - topic clustering` task. It is a **pipeline verification
fixture**, not a real corpus and not evidence of real-world or clinical
performance.

## Data contract

`fixture.json` contains exactly six fictional records:

| record_id | text | label |
|---|---|---:|
| t1 | red apple | 1 |
| t2 | green apple | 1 |
| t3 | blue sky | 0 |
| t4 | dark sky | 0 |
| t5 | red fruit | 1 |
| t6 | blue cloud | 0 |

The JSON top-level fields are `schema`, `simulated`, `description`, and
`records`. Every record must contain exactly `record_id`, `text`, and
`label`. IDs are strings, text is a string, and labels are integer `0` or `1`.
The fixture is explicitly marked `"simulated": true`.

## Pipeline

`pipeline.py` implements a standard-library-only multinomial Naive Bayes text
pipeline. The fixed seeded split (seed `42`) is:

- training: t4, t2, t3, t5
- validation: t1
- test: t6

The fixed model parameters and class statistics are stored in JSON. The public
functions validate the fixture/model schemas, save/load JSON models, and expose
`predict(model, texts)`. There is no pickle loading and inference does not need
training records or credentials.

## Run and verify

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t060 -p "test_*.py"
```

The tests use Python's standard library and fixed, CPU-only, six-record data.
They verify the literal fixture, split separation, save/reload prediction
agreement, schema failures, and inference without training records.

## Limits

This component does not download, use, or reproduce the real 20 Newsgroups
dataset. It does not provide corpus-level accuracy, clustering, topic quality,
or clinical claims. A real-data extension would require a separately documented
source, licence, access requirements, split design, and performance scope.
