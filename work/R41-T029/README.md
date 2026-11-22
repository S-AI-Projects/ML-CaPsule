# Verification Evidence — R41-T029

**Task:** R41-T029 · 9 of 10 (`Goodbooks-10k – content-based recommender` robustness fixture)  
**Student:** Sharon Robinson  
**Branch:** `R41-T029` (assigned working branch: `student/e26b07f0819`)

> **Note for upstream PRs:** Per task instructions, `work/R41-T029/` is retained on the organisation-fork coursework branch for review and must be removed from any later upstream branch.

## Artefacts

| File | Description |
|------|-------------|
| `inputs_fixture.json` | Exact six synthetic interactions `(u1,a,5),(u1,b,3),(u2,a,4),(u2,c,2),(u3,b,4),(u3,c,5)` and synthetic item catalog |
| `expected_vs_actual.json` | Expected vs actual aligned scores, top-2 rankings, order-invariance check (`invariance_preserved = true`, `max_abs_diff = 1.11e-16`), and explicit fallback outputs |
| `unittest_output.txt` | Full verbose output of `python3 -m unittest discover -s course-labs/r41-t029 -p "test_*.py" -v` (12 tests, exit code `0`) |
| `mutation_check.txt` | Deliberate mutation verification altering `EXPECTED_HELD_OUT_SCORES["u1:b"]` from `0.5` to `0.99` (exit code `1`, non-zero as required) |
| `tool_versions.txt` | Python, platform, and git versions |
| `pr_body.txt` | Mandatory PR description template prepared for the organisation-fork PR |

## Synthetic Data & Scope Notice

All outputs recorded in this folder come strictly from six synthetic interactions for **pipeline verification** — they are **not** observations from Goodbooks-10k, **not** evidence of real-world or clinical performance, and do **not** claim broad adversarial robustness.
