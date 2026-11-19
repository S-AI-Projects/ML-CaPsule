# Work Verification Evidence — Task R41-T021

This directory captures execution artifacts, diagnostic tests, environment versions, and mutation failure evidence for Task R41-T021: **MNIST – k-NN baseline (Validate data contract)**.

> **Notice:** As per task instructions, all records are synthetic CPU-only smoke verification fixtures, not observations from the real MNIST dataset. This directory is included for coursework verification on branch `R41-T021` and excluded from upstream proposals.

## Contents

- `inputs_fixture.json`: Copy of canonical synthetic fixture (`course-labs/r41-t021/fixture.json`) with six 4×4 images (3 all-zero, 3 white diagonal).
- `expected_vs_actual.json`: Comparison of expected schema validation flags, failure diagnoses, and 1-NN predictions against actual outputs.
- `tool_versions.txt`: Runtime tool and environment versions (Python, git, OS kernel/architecture).
- `unittest_output.txt`: Full terminal output from `python3 -m unittest discover -s course-labs/r41-t021 -p "test_*.py" -v` reporting 10 passing tests.
- `mutation_check.txt`: Verification log demonstrating that a mutated expected result triggers an `AssertionError` and exits with a non-zero code.
- `pr_body.txt`: Standard PR submission template formatted according to the project evaluation rubric.
