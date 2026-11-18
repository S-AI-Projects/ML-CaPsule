# R41-T044 check evidence

This coursework-only directory records the inputs, expected result, actual
output, mutation check, and runtime used for the offline image-metric fixture.
It is not part of a later upstream proposal.

The required check is:

```bash
python3 -m unittest discover -s course-labs/r41-t044 -p "test_*.py"
```

It completed with 10 tests and exit status 0. `mutation_check.txt` records a
deliberately incorrect precision expectation and the resulting non-zero Python
exit status.
