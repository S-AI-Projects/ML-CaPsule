# R41-T002: Deterministic Split Manifest

## Objective

Implement a deterministic, reproducible, and group-isolated train/validation/test split manifest for a synthetic classification fixture while strictly preventing group and record leakage.

> **DISCLAIMER:**
> All records in this fixture are simulated synthetic examples. They are **NOT** observations from a real Parkinson's disease dataset, clinical trial, or medical study, and must not be interpreted as evidence of clinical performance.

---

## Data Schema & Contract

Each record in the dataset is a dictionary conforming to the following schema:

| Field Name | Type | Constraints | Description |
|---|---|---|---|
| `record_id` | `str` | Required, non-empty, unique string | Stable record identifier |
| `group_id` | `str` | Required, non-empty string | Stable group identifier for group isolation |
| `x1` | `int` or `float` | Required, numeric scalar (0–7 in fixture) | First numerical feature |
| `x2` | `int` or `float` | Required, numeric scalar (constant 1 in fixture) | Second numerical feature |
| `label` | `int` | Required, binary integer: `0` or `1` | Binary classification target |

---

## Synthetic Fixture

The fixture consists of exactly eight synthetic records distributed across four groups, with each group containing one negative (`0`) and one positive (`1`) class observation:

| Group ID | Records | Labels |
|---|---|---|
| `grp_01` | `rec_001` (label 0), `rec_005` (label 1) | {0, 1} |
| `grp_02` | `rec_002` (label 0), `rec_006` (label 1) | {0, 1} |
| `grp_03` | `rec_003` (label 0), `rec_007` (label 1) | {0, 1} |
| `grp_04` | `rec_004` (label 0), `rec_008` (label 1) | {0, 1} |

---

## Split Strategy & Group-Isolation Guarantees

The splitting pipeline enforces the following guarantees:
1. **Deterministic & Seeded:** Uses a fixed random seed of `42` (`SPLIT_SEED = 42`) to deterministically shuffle groups.
2. **Whole Group Allocation:** Entire groups—never individual records—are assigned to splits.
3. **Partition Completeness:** Every record is assigned to exactly one split (train, validation, or test).
4. **No Record or Group Overlap:**
   - No record ID appears in more than one split.
   - No group ID appears in more than one split.
5. **Class Representation:** Because every group contains both classes ({0, 1}), each split retains representation of both labels.

### Resulting Manifest (Seed 42)

| Split | Assigned Groups | Record IDs | Record Count | Class Counts (0 : 1) |
|---|---|---|---:|---|
| **Train** | `grp_03`, `grp_02` | `rec_002`, `rec_003`, `rec_006`, `rec_007` | 4 | 2 : 2 (50% / 50%) |
| **Validation** | `grp_04` | `rec_004`, `rec_008` | 2 | 1 : 1 (50% / 50%) |
| **Test** | `grp_01` | `rec_001`, `rec_005` | 2 | 1 : 1 (50% / 50%) |

---

## Time and User Isolation Limitations

The synthetic schema contains no timestamp, chronological ordering, or patient/user identifier fields. Consequently:
- **Time-based isolation** is not exercised by this fixture.
- **User-based isolation** is not exercised by this fixture.
Group isolation is strictly enforced via `group_id`.

---

## How to Run the Tests

The test suite runs offline on CPU and uses only Python standard library modules (`unittest`, `random`, `pathlib`, `sys`).

From the repository root:

```bash
python3 -m unittest discover -s course-labs/r41-t002 -p "test_*.py" -v
```
