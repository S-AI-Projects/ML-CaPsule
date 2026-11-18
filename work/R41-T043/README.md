# R41-T043 verification evidence

The fixture inputs are recorded in `inputs.json`; the asserted result is in
`expected.json`, and the CLI output from `pipeline.py` is in `actual.json`.
They match exactly. These results are only pipeline verification for fictional
simulated records.

The required offline check was run from the repository root:

```text
python3 -m unittest discover -s course-labs/r41-t043 -p "test_*.py"
Ran 11 tests
OK
```

Tool and runtime versions are recorded in `tool_versions.txt`. The unittest
runner exits non-zero for a failed assertion; `mutation_check.txt` records the
deliberately failing assertion control.
