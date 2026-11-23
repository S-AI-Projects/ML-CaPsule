"""Moving-Average Forecast and Error-Analysis Pipeline for Task R41-T008.

Implements:
1. Moving average forecasting on sequential time series observations without shuffling.
2. Structured error-analysis reporting:
   - Stable record_id, expected value, predicted value, residual, and reason category.
   - Filters correct predictions (within tolerance) so only actual errors appear.
   - Aggregate error counts strictly match detailed error rows.
3. Injected wrong prediction mechanism for rigorous testability.
"""

import math
from typing import Any, Dict, List, Optional, Set

ID_FIELD: str = "record_id"
TIMESTAMP_FIELD: str = "timestamp"
VALUE_FIELD: str = "value"

DEFAULT_TOLERANCE: float = 2.0


class SchemaValidationError(TypeError):
    """Raised when record types or schemas violate the contract."""
    pass


def validate_time_series_records(records: List[Dict[str, Any]]) -> None:
    """Validate schema, chronologically ordered records, and unique IDs."""
    if not isinstance(records, list) or len(records) == 0:
        raise SchemaValidationError("Records must be a non-empty list of dicts")

    seen_ids: Set[str] = set()
    for rec in records:
        if not isinstance(rec, dict):
            raise SchemaValidationError("Each record must be a dict")
        if ID_FIELD not in rec:
            raise SchemaValidationError(f"Missing required field '{ID_FIELD}'")
        rid = rec[ID_FIELD]
        if not isinstance(rid, str) or not rid.strip():
            raise SchemaValidationError(f"'{ID_FIELD}' must be a non-empty string, got {rid!r}")
        if rid in seen_ids:
            raise ValueError(f"Duplicate record ID '{rid}' detected in time series")
        seen_ids.add(rid)

        if TIMESTAMP_FIELD not in rec:
            raise SchemaValidationError(f"Missing required field '{TIMESTAMP_FIELD}' in record '{rid}'")
        ts = rec[TIMESTAMP_FIELD]
        if not isinstance(ts, str) or not ts.strip():
            raise SchemaValidationError(f"'{TIMESTAMP_FIELD}' must be a non-empty string in record '{rid}'")

        if VALUE_FIELD not in rec:
            raise SchemaValidationError(f"Missing required field '{VALUE_FIELD}' in record '{rid}'")
        val = rec[VALUE_FIELD]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise SchemaValidationError(f"'{VALUE_FIELD}' must be numeric, got {type(val).__name__} in '{rid}'")
        if math.isnan(val) or math.isinf(val):
            raise SchemaValidationError(f"'{VALUE_FIELD}' in record '{rid}' must be a finite number")


class MovingAverageForecaster:
    """Sequential moving-average forecaster with rolling window k."""

    def __init__(self, window_size: int = 3) -> None:
        if window_size < 1:
            raise ValueError("window_size must be at least 1")
        self.window_size: int = window_size
        self.history_: List[float] = []

    def fit(self, train_records: List[Dict[str, Any]]) -> "MovingAverageForecaster":
        """Fit history sequentially from training records without shuffling."""
        validate_time_series_records(train_records)
        self.history_ = [float(r[VALUE_FIELD]) for r in train_records]
        return self

    def forecast_next(self) -> float:
        """Forecast the next value as the mean of the last window_size observations."""
        if len(self.history_) == 0:
            raise RuntimeError("Forecaster has no history to forecast from")
        k = min(self.window_size, len(self.history_))
        return sum(self.history_[-k:]) / k

    def step(self, actual_value: float) -> None:
        """Append an actual observed value to update rolling history."""
        self.history_.append(float(actual_value))


def categorize_error(residual: float, threshold_anomaly: float = 10.0) -> str:
    """Categorize reason for forecasting error."""
    if abs(residual) >= threshold_anomaly:
        return "INJECTED_ANOMALY"
    elif residual > 0:
        return "OVERESTIMATION"
    else:
        return "UNDERESTIMATION"


def generate_error_analysis_report(
    predictions_by_id: Dict[str, float],
    expected_records: List[Dict[str, Any]],
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, Any]:
    """Generate structured error-analysis report comparing predictions against expected records.

    Rules:
    - If abs(predicted - expected) <= tolerance, prediction is correct and does NOT appear in error_rows.
    - If abs(predicted - expected) > tolerance, error row is recorded with reason_category.
    - Aggregate error_count strictly matches len(error_rows).
    """
    validate_time_series_records(expected_records)

    error_rows: List[Dict[str, Any]] = []
    correct_count = 0

    for rec in expected_records:
        rid = rec[ID_FIELD]
        if rid not in predictions_by_id:
            raise ValueError(f"No prediction provided for record_id '{rid}'")

        expected = float(rec[VALUE_FIELD])
        predicted = float(predictions_by_id[rid])
        residual = predicted - expected
        abs_err = abs(residual)

        if abs_err > tolerance:
            reason = categorize_error(residual)
            error_rows.append({
                "record_id": rid,
                "expected": expected,
                "predicted": predicted,
                "residual": residual,
                "reason_category": reason,
            })
        else:
            correct_count += 1

    total_evaluated = len(expected_records)
    error_count = len(error_rows)

    assert error_count + correct_count == total_evaluated, "Counts must sum to total evaluated"

    return {
        "total_evaluated": total_evaluated,
        "error_count": error_count,
        "correct_count": correct_count,
        "tolerance": tolerance,
        "error_rows": error_rows,
    }


def create_predictions_with_injected_error(
    expected_records: List[Dict[str, Any]],
    injected_wrong_id: str,
    injected_wrong_value: float = 99.0,
) -> Dict[str, float]:
    """Construct predictions matching expected values except for one deliberately injected wrong ID."""
    predictions: Dict[str, float] = {}
    found_injected = False

    for rec in expected_records:
        rid = rec[ID_FIELD]
        if rid == injected_wrong_id:
            predictions[rid] = float(injected_wrong_value)
            found_injected = True
        else:
            predictions[rid] = float(rec[VALUE_FIELD])

    if not found_injected:
        raise ValueError(f"Injected ID '{injected_wrong_id}' not found in expected records")

    return predictions
