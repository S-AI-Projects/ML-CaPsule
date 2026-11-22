# R41-T041 check evidence

The saved input, expected report, actual report, and runtime versions are in
this directory. The actual result was captured from the required offline test
run: 10 tests, all passing. The fixture and its split IDs are simulated and
deterministic; results are pipeline verification only, not real-world concrete
strength performance evidence.

The suite includes assertions for valid and invalid records. Its mutation-sanity
test deliberately compares the actual record count with an incorrect expected
count and confirms Python raises `AssertionError`; changing a real expected
value in a test therefore fails unittest and produces a non-zero process exit.

Do not include this `work/R41-T041/` evidence directory in any later upstream
proposal branch.