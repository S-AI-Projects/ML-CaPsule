# work/R41-T023 – Evidence Directory

## Task
**R41-T023 · 3 of 10** — Auto MPG – regression (minimal baseline comparison)
**Student:** Sharon Robinson | **Branch:** R41-T023

## Contents

| File | Description |
|------|-------------|
| `inputs_fixture.json` | Copy of `course-labs/r41-t023/fixture.json` — 6-row synthetic formula fixture |
| `expected_vs_actual.json` | Recorded baseline parameters and evaluation metrics (MAE, MSE, RMSE) |
| `tool_versions.txt` | Python, git and OS versions at run-time |
| `unittest_output.txt` | Captured output of `python3 -m unittest discover -s course-labs/r41-t023 -p "test_*.py" -v` — 10 tests OK |
| `mutation_check.txt` | Deliberate-mutation check output — confirms assertion fails (exit 1) on wrong expected value |
| `pr_body.txt` | PR description submitted to upstream |

## Verification Command

```bash
python3 -m unittest discover -s course-labs/r41-t023 -p "test_*.py" -v
```

Expected: `Ran 10 tests in ~0.001s ... OK`

## Key Results

| Split | MAE | MSE | RMSE |
|-------|-----|-----|------|
| Validation (r5) | 5.0 | 25.0 | 5.0 |
| Test (r6) | 7.0 | 49.0 | 7.0 |

- Training mean = **4.0** (computed from targets `[1, 3, 5, 7]`)
- Oracle contrast (LinearRegressionComparator): slope=2.0, intercept=1.0, validation MSE=0.0

## Data Disclaimer
All fixture rows are synthetic (formula `target = 2*x1 + 1`). These are not observations from the named real Auto MPG dataset.
