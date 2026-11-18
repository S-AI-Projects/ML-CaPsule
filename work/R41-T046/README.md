# R41-T046 verification evidence

The eight simulated input rows are recorded in `inputs.json`; the asserted
result is in `expected.json`, and the CLI output is in `actual.json`. The files
match exactly. The baseline uses zero update steps and reports finite
probabilities and binary cross-entropy.

The required offline check was run from the repository root:

```text
python3 -m unittest discover -s course-labs/r41-t046 -p "test_*.py"
Ran 8 tests
OK
```

Tool/runtime versions are in `tool_versions.txt`. The check is CPU-only,
offline, and bounded to eight rows. `mutation_check.txt` records a deliberately
failed expected-value control. Results are pipeline smoke-test verification,
not Taiwan credit-default performance evidence.
