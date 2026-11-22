# Verification Evidence — R41-T028

**Task:** R41-T028 · 8 of 10 (`Twitter airline sentiment – Naive Bayes` error-analysis report)  
**Student:** Sharon Robinson  
**Branch:** `R41-T028` (assigned working branch: `student/e26b07f0819`)

> **Note for upstream PRs:** Per task instructions, `work/R41-T028/` is retained on the organisation-fork coursework branch for review and must be removed from any later upstream branch.

## Artefacts

| File | Description |
|------|-------------|
| `inputs_fixture.json` | Exact six fictional text records (`t1..t6`, labels `[1,1,0,0,1,0]`) |
| `expected_vs_actual.json` | Expected vs actual Naive Bayes predictions, injected error report (`t5` present, correct predictions excluded), and aggregate reconciliation (`all_match = true`) |
| `unittest_output.txt` | Full verbose output of `python3 -m unittest discover -s course-labs/r41-t028 -p "test_*.py" -v` (17 tests, exit code `0`) |
| `mutation_check.txt` | Deliberate mutation verification altering `EXPECTED_INJECTED_ERROR_COUNT` from `1` to `2` (exit code `1`, non-zero as required) |
| `tool_versions.txt` | Python, platform, and git versions |
| `pr_body.txt` | Mandatory PR description template prepared for the organisation-fork PR |

## Synthetic Data Notice

All outputs recorded in this folder come strictly from six fictional two-word text records (`t1..t6`) for **pipeline verification** and are **not** a real Twitter airline sentiment corpus or evidence of real-world or clinical performance.
