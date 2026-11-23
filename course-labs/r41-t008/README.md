# R41-T008: Error-Analysis Example (Air Passengers – Moving-Average Forecast)

## Objective
Implement a structured time series error-analysis report for a sequential moving-average forecast fixture:
1. Records evaluation details: stable record ID, expected value, predicted value, residual, and a classified reason category (`INJECTED_ANOMALY`, `OVERESTIMATION`, `UNDERESTIMATION`).
2. Validates error filtering: accurate predictions within tolerance do not appear in the detailed error rows, while a deliberately injected wrong prediction appears with its exact record ID.
3. Guarantees that aggregate error and correct counts strictly match the detailed report rows.

## Synthetic Fixture Specification
> [!IMPORTANT]
> **DISCLAIMER:** This time series dataset is an explicitly synthetic, deterministic verification fixture created solely for offline pipeline verification and error-analysis testing. It does **NOT** contain real-world Box-Jenkins airline passenger counts, international flight traffic, or real-world aviation data.

### Schema
| Field | Type | Description |
|---|---|---|
| `record_id` | String | Unique chronological ID (`d1`..`d10`) |
| `timestamp` | String | Daily ISO timestamp (`2026-01-01`..`2026-01-10`) |
| `value` | Numeric | Observation `[10, 12, 11, 13, 15, 14, 16, 18, 17, 19]` |

### Sequential Split Manifest (No Shuffling)
- **Training (first 6 rows):** `["d1", "d2", "d3", "d4", "d5", "d6"]`
- **Validation (next 2 rows):** `["d7", "d8"]`
- **Test (final 2 rows):** `["d9", "d10"]`

## Error-Analysis Structure
```json
{
  "total_evaluated": 4,
  "error_count": 1,
  "correct_count": 3,
  "tolerance": 2.0,
  "error_rows": [
    {
      "record_id": "d10",
      "expected": 19.0,
      "predicted": 99.0,
      "residual": 80.0,
      "reason_category": "INJECTED_ANOMALY"
    }
  ]
}
```

## Test Verification
Run the offline unit tests:
```bash
python3 -m unittest discover -s course-labs/r41-t008 -p "test_*.py" -v
```
All 7 tests execute offline in <0.01 seconds using Python standard library only.
