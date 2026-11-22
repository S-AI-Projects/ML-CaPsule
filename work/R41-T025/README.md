# Verification Evidence — R41-T025

**Task:** R41-T025 · 5 of 10 (`Zoo animals – decision tree` numeric preprocessing)  
**Student:** Sharon Robinson  
**Branch:** `R41-T025` (assigned working branch: `student/e26b07f0819`)

> **Note for upstream PRs:** Per task instructions, `work/R41-T025/` is retained on the organisation-fork coursework branch for review and must be removed from any later upstream branch.

## Artefacts

| File | Description |
|------|-------------|
| `inputs_fixture.json` | Exact eight-row synthetic input fixture (`x1=[0..7]`, `x2=1`, `label=[0,0,0,0,1,1,1,1]`) |
| `expected_vs_actual.json` | Expected vs actual fitted statistics and transformed `float64` outputs across `train`, `validation`, and `test` (`max_abs_diff = 0.0`) |
| `unittest_output.txt` | Full verbose output of `python3 -m unittest discover -s course-labs/r41-t025 -p "test_*.py" -v` (28 tests, exit code `0`) |
| `mutation_check.txt` | Deliberate mutation verification changing expected training mean `2.5 -> 3.5` (exit code `1`, non-zero as required) |
| `tool_versions.txt` | Python, platform, git, and optional cross-check library versions |
| `pr_body.txt` | Mandatory PR description template prepared for the organisation-fork PR |

## Synthetic Data Notice

All outputs recorded in this folder come exclusively from the eight-row synthetic fixture for **pipeline verification** and are **not** observations from the UCI Zoo dataset or evidence of real-world or clinical performance.
