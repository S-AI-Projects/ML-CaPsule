# Verification Evidence — R41-T026

**Task:** R41-T026 · 6 of 10 (`Building energy efficiency – regression` bounded CPU smoke test)  
**Student:** Sharon Robinson  
**Branch:** `R41-T026` (assigned working branch: `student/e26b07f0819`)

> **Note for upstream PRs:** Per task instructions, `work/R41-T026/` is retained on the organisation-fork coursework branch for review and must be removed from any later upstream branch.

## Artefacts

| File | Description |
|------|-------------|
| `inputs_fixture.json` | Exact six-row synthetic deterministic formula fixture (`x1=[0..5]`, `x2=1`, `target=2*x1+1`, train `r1..r4`, validation `r5`, test `r6`) |
| `expected_vs_actual.json` | Expected vs actual smoke-run parameters, prediction shapes, finite losses, and predictions (`max_abs_diff = 1.11e-16`) |
| `unittest_output.txt` | Full verbose output of `python3 -m unittest discover -s course-labs/r41-t026 -p "test_*.py" -v` (20 tests, exit code `0`) |
| `mutation_check.txt` | Deliberate mutation verification altering `EXPECTED_SGD_W1` from `2.02` to `9.99` (exit code `1`, non-zero as required) |
| `tool_versions.txt` | Python, platform, and git versions |
| `pr_body.txt` | Mandatory PR description template prepared for the organisation-fork PR |

## Synthetic Data Notice

All outputs recorded in this folder come strictly from the six-row deterministic formula fixture for **pipeline verification** and are **not** observations from the UCI Energy Efficiency dataset or evidence of real-world or clinical performance.
