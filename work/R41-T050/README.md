# R41-T050 evidence

## Fixture

Ten literal simulated daily observations are recorded with IDs d1-d10 and values
`[10,12,11,13,15,14,16,18,17,19]`. IDs d1-d6 are train, d7-d8 validation, and
d9-d10 test. There is no shuffle and timestamps are consecutive.

## Saved-model result

The train-only median is `12.5`. Saving to JSON and reloading produces the same
configuration and `predict` result within the explicit tolerance `1e-12`.
Inference receives only the prediction record and parameters; no training data,
credentials, NumPy, or pickle load is used.

## Mutation proof

Changing the final fixture value in an assertion to `19.1` made the unittest
check exit 1. The mutated assertion was restored afterward. Machine-readable
inputs, expected/actual values, tool versions, normal output, mutated output,
and mutation exit code are stored alongside this file.

The data is pipeline-verification evidence only, not real Indian-rainfall or
clinical-performance evidence.
