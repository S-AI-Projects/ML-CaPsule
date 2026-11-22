# R41-T049 evidence

## Deterministic fixture

The check uses six literal simulated rows with `x1 = [0,1,2,3,4,5]`, `x2 = 1`,
and `target = 2*x1+1`. Stable IDs are r1 through r6. Train is r1-r4, validation
is r5, and test is r6. The seed is fixed at 49; no time order or random split is
used.

## Expected versus actual

- Expected and actual coefficients: `[1.0, 2.0]`
- Validation r5: target `9.0`, predicted `9.0`, absolute error `0.0`
- Test r6: target `11.0`, predicted `11.0`, absolute error `0.0`
- Reversing record order produced the same result by stable-ID alignment.

Machine-readable inputs, expected values, actual values, and tool versions are
saved alongside this file. The normal check reports 6 tests and `OK`.

## Mutation proof

A deliberate seed assertion mutation (49 changed to 50) made the required unit
check exit 1, and the output is saved in `mutated_assertion_output.txt`.
The assertion was restored after the mutation run.

These outcomes verify the bounded synthetic pipeline only. They are not evidence
of Avocado-price, real-world, or clinical performance.
