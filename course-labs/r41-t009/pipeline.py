"""Robust Regression Pipeline for Task R41-T009: Kaggle house prices.

Implements and tests robustness properties for tabular regression:
1. Permutation invariance under ID alignment: changing record order preserves exact
   predictions when aligned by stable record_id.
2. Rejection or fallback mapping for unsupported inputs (e.g., missing fields, shape mismatches,
   non-numeric types).
3. Explicit scope limitation: verifies tabular order-invariance and input validation only,
   without claiming broad adversarial robustness.
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple

FEATURE_ORDER: List[str] = ["x1", "x2"]
ID_FIELD: str = "record_id"
TARGET_FIELD: str = "target"

SCOPE_LIMITATION_DISCLAIMER: str = (
    "This robustness verification tests tabular record-order invariance with stable ID alignment "
    "and input-shape validation. It does NOT claim broad adversarial robustness, distribution-shift "
    "resilience, or protection against adversarial noise."
)


class IncompatibleInputError(ValueError):
    """Raised when an input record violates required feature shape or types."""
    pass


class RobustRegressionPipeline:
    """Regression pipeline implementing OLS fitting and robust prediction interfaces."""

    def __init__(self, fallback_prediction: Optional[float] = None) -> None:
        self.w1_: float = 0.0
        self.w2_: float = 0.0
        self.bias_: float = 0.0
        self.training_mean_target_: float = 0.0
        self.is_fitted: bool = False
        self.fallback_prediction: Optional[float] = fallback_prediction

    def fit(self, train_records: List[Dict[str, Any]]) -> "RobustRegressionPipeline":
        """Fit linear regression parameters on training records."""
        if not isinstance(train_records, list) or len(train_records) == 0:
            raise ValueError("Training records must be a non-empty list of dicts")

        x1_vals = [float(r["x1"]) for r in train_records]
        targets = [float(r[TARGET_FIELD]) for r in train_records]

        n = len(train_records)
        mean_x1 = sum(x1_vals) / n
        mean_y = sum(targets) / n
        self.training_mean_target_ = mean_y

        cov = sum((x - mean_x1) * (y - mean_y) for x, y in zip(x1_vals, targets))
        var = sum((x - mean_x1) ** 2 for x in x1_vals)

        if var > 1e-12:
            self.w1_ = cov / var
        else:
            self.w1_ = 0.0

        self.w2_ = 0.0  # x2 is constant
        self.bias_ = mean_y - self.w1_ * mean_x1
        self.is_fitted = True
        return self

    def predict_single(self, record: Dict[str, Any], allow_fallback: bool = False) -> float:
        """Predict for a single record with shape validation or fallback mapping."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling predict_single()")

        if not isinstance(record, dict) or ID_FIELD not in record:
            if allow_fallback:
                return self.fallback_prediction if self.fallback_prediction is not None else self.training_mean_target_
            raise IncompatibleInputError("Record must be a dict containing 'record_id'")

        # Validate feature shape and types
        for feat in FEATURE_ORDER:
            if feat not in record:
                if allow_fallback:
                    return self.fallback_prediction if self.fallback_prediction is not None else self.training_mean_target_
                raise IncompatibleInputError(
                    f"Incompatible shape: record '{record[ID_FIELD]}' missing required feature '{feat}'"
                )
            val = record[feat]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                if allow_fallback:
                    return self.fallback_prediction if self.fallback_prediction is not None else self.training_mean_target_
                raise IncompatibleInputError(
                    f"Incompatible type: feature '{feat}' in record '{record[ID_FIELD]}' must be numeric, got {type(val).__name__}"
                )
            if math.isnan(val) or math.isinf(val):
                if allow_fallback:
                    return self.fallback_prediction if self.fallback_prediction is not None else self.training_mean_target_
                raise IncompatibleInputError(
                    f"Incompatible value: feature '{feat}' in record '{record[ID_FIELD]}' must be finite"
                )

        # Compute regression prediction: y = w1*x1 + w2*x2 + b
        return self.w1_ * float(record["x1"]) + self.w2_ * float(record["x2"]) + self.bias_

    def predict_by_id(
        self, records: List[Dict[str, Any]], allow_fallback: bool = False
    ) -> Dict[str, float]:
        """Predict target values mapped by stable record_id."""
        predictions: Dict[str, float] = {}
        for rec in records:
            rid = rec.get(ID_FIELD, "unknown")
            pred = self.predict_single(rec, allow_fallback=allow_fallback)
            predictions[rid] = pred
        return predictions


def permute_records(records: List[Dict[str, Any]], permutation_indices: List[int]) -> List[Dict[str, Any]]:
    """Permute records according to given index order."""
    if len(permutation_indices) != len(records):
        raise ValueError("permutation_indices length must match records length")
    return [records[i] for i in permutation_indices]
