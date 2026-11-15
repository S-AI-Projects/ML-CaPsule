# R41-T001: Data Contract Validation — IPL Matches Win Predictor

## Goal

This component implements a strict schema validator for classification input records associated with the *IPL matches – EDA + win predictor* workflow. It ensures that input datasets strictly conform to the expected data contract prior to feature engineering or model training, and provides granular failure reporting without silently discarding records.

> **DISCLAIMER:**  
> The fixture data provided and tested in this directory consists exclusively of eight synthetic records generated for pipeline verification and contract testing. These records do **NOT** represent real IPL match observations, ball-by-ball deliveries, or win probabilities, and must not be used as evidence of real-world model performance.

---

## Data Contract Specification

Each candidate record must be a dictionary containing the following fields:

| Field Name | Type | Constraints & Validation Rules | Description |
|---|---|---|---|
| `record_id` | `str` | Required, non-empty, non-whitespace string; unique across the dataset | Stable row identifier for tracking records |
| `group_id` | `str` | Required, non-empty, non-whitespace string | Stable group/cluster identifier |
| `x1` | `int` or `float` | Required, scalar numeric value (finite real number); booleans, sequences, and tensors strictly rejected | First numerical feature |
| `x2` | `int` or `float` | Required, scalar numeric value (finite real number); booleans, sequences, and tensors strictly rejected | Second numerical feature |
| `label` | `int` | Required binary integer; strictly `0` or `1`; booleans, floats, and out-of-bounds integers rejected | Binary target outcome |

### Error Reporting (No Silent Dropping)
When a record violates one or more rules, the validator captures **every** failure without dropping the record:
- `record_id`: The identifier of the failing record (or `<missing>` / `<empty>` if absent).
- `field`: The name of the invalid field.
- `reason`: A human-readable description of the constraint violation.

---

## Synthetic Fixture Specification

The canonical synthetic fixture consists of exactly eight rows:

| record_id | group_id | x1 | x2 | label |
|---|---|---|---|---|
| `rec_001` | `grp_01` | 0 | 1 | 0 |
| `rec_002` | `grp_01` | 1 | 1 | 0 |
| `rec_003` | `grp_01` | 2 | 1 | 0 |
| `rec_004` | `grp_01` | 3 | 1 | 0 |
| `rec_005` | `grp_02` | 4 | 1 | 1 |
| `rec_006` | `grp_02` | 5 | 1 | 1 |
| `rec_007` | `grp_02` | 6 | 1 | 1 |
| `rec_008` | `grp_02` | 7 | 1 | 1 |

### Fixed Stratified Split Design

To demonstrate train/test separation without data leakage while preserving class balance:
- **Train split (6 records, 75%)**: `rec_001`, `rec_002`, `rec_003` (label 0) and `rec_005`, `rec_006`, `rec_007` (label 1). Class balance: 3 negative, 3 positive (50% / 50%).
- **Test split (2 records, 25%)**: `rec_004` (label 0) and `rec_008` (label 1). Class balance: 1 negative, 1 positive (50% / 50%).

---

## How to Run the Tests

The test suite uses Python's built-in `unittest` framework and has **zero external dependencies** (standard library only).

Run the automated check from the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t001 -p "test_*.py" -v
```

---

## Scope and Limitations

- **Bounded Component**: This lab implements the data contract validator and synthetic fixture for task R41-T001. It is not a complete reproduction of the full IPL match win predictor notebook.
- **CPU-Only & Offline**: Does not require GPUs, internet connectivity, or external packages (such as NumPy, PyTorch, or Pandas).
- **Synthetic Purpose**: The 8 fixture records serve strictly for input contract verification. Any future real-data extension must separately record licensing, data sources, and split designs.

