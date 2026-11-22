# Verification Evidence — R41-T027

**Task:** R41-T027 · 7 of 10 (`Banknote authentication – SVM` reproducible configuration)  
**Student:** Sharon Robinson  
**Branch:** `R41-T027` (assigned working branch: `student/e26b07f0819`)

> **Note for upstream PRs:** Per task instructions, `work/R41-T027/` is retained on the organisation-fork coursework branch for review and must be removed from any later upstream branch.

## Artefacts

| File | Description |
|------|-------------|
| `inputs_fixture.json` | Exact eight-row synthetic classification fixture (`x1=[0..7]`, `x2=1`, `label=[0,0,0,0,1,1,1,1]`) |
| `inputs_config.json` | Serializable experiment configuration (`schema_version="1.0.0"`, `seed=42`, `feature_ordering`, `split_ids`, `algorithm`) |
| `expected_vs_actual.json` | Expected vs actual parameters, splits, predictions, round-trip check (`roundtrip_identical=true`, `max_abs_diff=0.0`), and environment versions |
| `unittest_output.txt` | Full verbose output of `python3 -m unittest discover -s course-labs/r41-t027 -p "test_*.py" -v` (18 tests, exit code `0`) |
| `mutation_check.txt` | Deliberate mutation verification altering `EXPECTED_VAL_PREDS` from `[0, 1]` to `[1, 0]` (exit code `1`, non-zero as required) |
| `tool_versions.txt` | Python, platform, git, and config schema versions |
| `pr_body.txt` | Mandatory PR description template prepared for the organisation-fork PR |

## Synthetic Data Notice

All outputs recorded in this folder come strictly from the eight-row synthetic fixture for **pipeline verification** and are **not** observations from the UCI Banknote Authentication dataset or evidence of real-world or clinical performance.
