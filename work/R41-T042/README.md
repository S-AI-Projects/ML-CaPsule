# R41-T042 check evidence

This folder contains the literal check inputs, expected manifest, and observed
manifest for the synthetic pipeline fixture. The matching expected/actual
outputs confirm the exact deterministic ID assignment for seed `41`.

Run the check from the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t042 -p "test_*.py"
```

The recorded result was 8 tests, exit status 0. A negative-control run altered
the expected IDs in memory; the exact-result test failed (1 failure, 7 passing)
and the wrapper confirmed failure detection. Tool details are in
[tool_versions.txt](tool_versions.txt). This evidence is pipeline verification
on fictional data and is not a real-world performance result.