# Verification Evidence — R41-T030

**Task:** R41-T030 · 10 of 10 (`Global CO2 emissions – trend forecast` saved-model interface)  
**Student:** Sharon Robinson  
**Branch:** `R41-T030` (assigned working branch: `student/e26b07f0819`)

> **Note for upstream PRs:** Per task instructions, `work/R41-T030/` is retained on the organisation-fork coursework branch for review and must be removed from any later upstream branch.

## Artefacts

| File | Description |
|------|-------------|
| `inputs_fixture.json` | Exact ten consecutive daily synthetic observations (`d1..d10`, values `[10,12,11,13,15,14,16,18,17,19]`) |
| `saved_model.json` | Safe JSON saved trend artifact (`schema_version="1.0.0"`, `intercept=72/7`, `slope=31/35`, no training data, no pickle) |
| `expected_vs_actual.json` | Expected vs actual parameters and before/after reload forecasts (`max_reload_abs_diff = 0.0`, `max_abs_diff = 1.78e-15`) |
| `unittest_output.txt` | Full verbose output of `python3 -m unittest discover -s course-labs/r41-t030 -p "test_*.py" -v` (12 tests, exit code `0`) |
| `mutation_check.txt` | Deliberate mutation verification altering `EXPECTED_SLOPE` from `31.0/35.0` to `5.0` (exit code `1`, non-zero as required) |
| `tool_versions.txt` | Python, platform, git, and artifact format versions |
| `pr_body.txt` | Mandatory PR description template prepared for the organisation-fork PR |

## Synthetic Data Notice

All outputs recorded in this folder come strictly from ten synthetic daily values (`d1..d10`) for **pipeline verification** — they are **not** real Global CO2 emissions observations or evidence of real-world or clinical performance.
