# R41-T007: Record Reproducible Configuration (Spotify Tracks – Genre Classification)

## Objective
Implement a strictly reproducible, JSON-serializable configuration system for a classification pipeline:
1. Records complete pipeline settings: schema version, random seed, feature ordering, split IDs, and algorithm hyperparameters.
2. Reconstructs identical dataset split partitioning and deterministic predictions upon configuration reloading.
3. Rejects unknown configuration fields, contradictory settings (e.g., overlapping split IDs, nonexistent record IDs, invalid hyperparameter ranges), and incompatible record input shapes.

## Synthetic Fixture Specification
> [!IMPORTANT]
> **DISCLAIMER:** This dataset is an explicitly synthetic verification fixture created solely for offline pipeline verification, configuration serialization testing, and reproducibility checks. It does **NOT** contain real-world Spotify audio features, acoustic track metadata, or actual music genre labels.

### Schema
| Field | Type | Description |
|---|---|---|
| `record_id` | String | Unique record identifier (`rec_001`..`rec_008`) |
| `group_id` | String | Group identifier (`grp_01`..`grp_04`) |
| `x1` | Numeric | Feature `[0, 1, 2, 3, 4, 5, 6, 7]` |
| `x2` | Numeric | Constant feature `[1, 1, 1, 1, 1, 1, 1, 1]` |
| `label` | Integer | Binary classification label `[0, 0, 0, 0, 1, 1, 1, 1]` |

### Fixed Stratified Split
- **Training (4 rows):** `["rec_001", "rec_002", "rec_005", "rec_006"]` (50% positive)
- **Validation (2 rows):** `["rec_003", "rec_007"]` (50% positive)
- **Test (2 rows):** `["rec_004", "rec_008"]` (50% positive)

## Configuration Specification
```json
{
  "schema_version": "1.0.0",
  "seed": 42,
  "feature_ordering": ["x1", "x2"],
  "split_ids": {
    "train": ["rec_001", "rec_002", "rec_005", "rec_006"],
    "validation": ["rec_003", "rec_007"],
    "test": ["rec_004", "rec_008"]
  },
  "algorithm_settings": {
    "algorithm": "logistic_regression",
    "learning_rate": 0.1,
    "max_iter": 50,
    "threshold": 0.5
  }
}
```

## Test Verification
Run the offline unit tests:
```bash
python3 -m unittest discover -s course-labs/r41-t007 -p "test_*.py" -v
```
All 12 tests execute offline in <0.01 seconds using Python standard library only.
