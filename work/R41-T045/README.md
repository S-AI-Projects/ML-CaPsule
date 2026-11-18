# R41-T045 verification evidence

The inputs are in `inputs.json`; the expected pipeline result is in
`expected.json`, and the CLI result is in `actual.json`. They match exactly.
All outputs are preprocessing verification for the explicitly simulated
eight-row fixture, not Iris-species performance.

The required offline check was run from the repository root:

```text
python3 -m unittest discover -s course-labs/r41-t045 -p "test_*.py"
Ran 12 tests
OK
```

`tool_versions.txt` records the runtime. The check is CPU-only, bounded to
eight rows, and uses no third-party packages. `mutation_check.txt` records a
deliberately changed expected value being rejected by the same test suite.
