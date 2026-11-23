# R41-T010: Saved-Pipeline Interface (Stroke Prediction – Class Imbalance with SMOTE)

## Objective
Implement a safe, serializable saved-pipeline interface for a classification model:
1. **Safe Serialization:** Saves and reloads model architecture, feature schema, and fitted parameters via standard JSON (explicitly avoiding untrusted pickle loading).
2. **Deterministic Agreement:** Before and after predictions (probabilities and predicted classes) agree within an explicit numerical tolerance ($10^{-7}$).
3. **Decoupled Inference:** Loaded model performs inference with **zero training data** and **zero external credentials**.
4. **Strict Schema & Version Validation:** Clear, descriptive exceptions for format-version mismatches (`SchemaVersionMismatchError`) and corrupted input record shapes (`SchemaValidationError`).

## Synthetic Fixture Specification
> [!IMPORTANT]
> **DISCLAIMER:** This dataset is an explicitly synthetic verification fixture created solely for offline pipeline verification, saved-model interface testing, and serialization reproducibility. It does **NOT** contain real-world clinical stroke observations, patient health records, or physiological measurements.

### Schema
| Field | Type | Description |
|---|---|---|
| `record_id` | String | Unique record ID (`rec_001`..`rec_008`) |
| `group_id` | String | Group identifier (`grp_01`..`grp_04`) |
| `x1` | Numeric | Feature `[0, 1, 2, 3, 4, 5, 6, 7]` |
| `x2` | Numeric | Constant feature `[1, 1, 1, 1, 1, 1, 1, 1]` |
| `label` | Integer | Binary classification label `[0, 0, 0, 0, 1, 1, 1, 1]` |

### Fixed Stratified Split
- **Training (4 rows):** `["rec_001", "rec_002", "rec_005", "rec_006"]` (50% positive)
- **Validation (2 rows):** `["rec_003", "rec_007"]` (50% positive)
- **Test (2 rows):** `["rec_004", "rec_008"]` (50% positive)

## Saved Model Structure
```json
{
  "format_version": "1.0.0",
  "model_type": "logistic_regression",
  "schema": {
    "id_field": "record_id",
    "features": ["x1", "x2"],
    "label_field": "label"
  },
  "parameters": {
    "weights": {
      "x1": 1.258,
      "x2": -0.891
    },
    "bias": -0.425,
    "threshold": 0.5
  }
}
```

## Test Verification
Run the offline unit tests:
```bash
python3 -m unittest discover -s course-labs/r41-t010 -p "test_*.py" -v
```
All 9 tests execute offline in <0.01 seconds using Python standard library only.
