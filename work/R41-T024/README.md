# work/R41-T024 – Evidence Directory

## Task
**R41-T024 · 4 of 10** — Loan Approval – Decision Tree: Classification Evaluation
**Student:** Sharon Robinson | **Branch:** R41-T024

## Contents

| File | Description |
|------|-------------|
| `inputs_fixture.json` | Copy of `course-labs/r41-t024/fixture.json` — 8-row synthetic classification fixture |
| `expected_vs_actual.json` | Recorded metric results for canonical and edge cases |
| `tool_versions.txt` | Python, git and OS versions at run-time |
| `unittest_output.txt` | Captured output of `python3 -m unittest discover -s course-labs/r41-t024 -p "test_*.py" -v` — 10 tests OK |
| `mutation_check.txt` | Deliberate-mutation check — confirms assertion exits non-zero when expected precision is changed |
| `pr_body.txt` | PR description submitted to upstream |

## Verification Command

```bash
python3 -m unittest discover -s course-labs/r41-t024 -p "test_*.py" -v
```

Expected: `Ran 10 tests in ~0.010s ... OK`

## Key Results

### Canonical Example
truth=[1,1,0,0] / prediction=[1,0,1,0]

| Metric | Value |
|--------|-------|
| TP / FP / FN / TN | 1 / 1 / 1 / 1 |
| Precision (binary) | **0.5** |
| Recall (binary) | **0.5** |
| F1 (binary) | **0.5** |

### Zero-Denominator
| Policy | Precision (all-negative predictions) |
|--------|--------------------------------------|
| `zero` | 0.0 |
| `nan` | NaN |

## Data Disclaimer
All fixture rows are synthetic. These are not observations from the named real Loan Approval dataset.
