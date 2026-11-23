"""Saved-Pipeline Interface for Task R41-T010: Stroke prediction – class imbalance.

Implements:
1. Safe JSON-based model serialization and deserialization (NO untrusted pickle loading).
2. Schema and format-version validation with explicit error handling.
3. Prediction interface requiring zero training data and zero external credentials at inference time.
4. Numerical verification that before/after predictions agree within an explicit tolerance.
"""

import json
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

SUPPORTED_FORMAT_VERSIONS: Set[str] = {"1.0.0"}
FEATURE_ORDER: List[str] = ["x1", "x2"]
ID_FIELD: str = "record_id"
LABEL_FIELD: str = "label"
DEFAULT_TOLERANCE: float = 1e-7


class SchemaVersionMismatchError(ValueError):
    """Raised when loaded model artifact version is unsupported."""
    pass


class SchemaValidationError(ValueError):
    """Raised when input records violate required schema or feature types."""
    pass


def validate_inference_record(record: Dict[str, Any], expected_features: List[str]) -> None:
    """Validate a single record against schema requirements for inference."""
    if not isinstance(record, dict):
        raise SchemaValidationError(f"Record must be a dict, got {type(record).__name__}")

    if ID_FIELD not in record:
        raise SchemaValidationError(f"Record missing required identifier '{ID_FIELD}'")

    rid = record[ID_FIELD]
    if not isinstance(rid, str) or not rid.strip():
        raise SchemaValidationError(f"'{ID_FIELD}' must be a non-empty string, got {rid!r}")

    for feat in expected_features:
        if feat not in record:
            raise SchemaValidationError(f"Record '{rid}' is missing required feature '{feat}'")
        val = record[feat]
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise SchemaValidationError(
                f"Feature '{feat}' in record '{rid}' must be numeric (int/float), got {type(val).__name__}"
            )
        if math.isnan(val) or math.isinf(val):
            raise SchemaValidationError(f"Feature '{feat}' in record '{rid}' must be finite")


class SavedClassificationPipeline:
    """Classification pipeline with safe JSON serialization and zero-dependency inference."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        bias: float = 0.0,
        threshold: float = 0.5,
        features: Optional[List[str]] = None,
        format_version: str = "1.0.0",
    ) -> None:
        self.format_version = format_version
        self.features = list(features or FEATURE_ORDER)
        self.weights_: Dict[str, float] = dict(weights or {})
        self.bias_: float = bias
        self.threshold: float = threshold
        self.is_fitted: bool = bool(weights)

    def fit(
        self,
        train_records: List[Dict[str, Any]],
        seed: int = 42,
        learning_rate: float = 0.1,
        max_iter: int = 60,
        threshold: float = 0.5,
    ) -> "SavedClassificationPipeline":
        """Fit binary logistic model on training records."""
        if not isinstance(train_records, list) or len(train_records) == 0:
            raise ValueError("train_records must be a non-empty list of dicts")

        for rec in train_records:
            validate_inference_record(rec, self.features)
            if LABEL_FIELD not in rec or rec[LABEL_FIELD] not in (0, 1):
                raise ValueError(f"Training record '{rec.get(ID_FIELD)}' must contain binary label in {{0, 1}}")

        rng = random.Random(seed)
        self.weights_ = {f: rng.uniform(-0.05, 0.05) for f in self.features}
        self.bias_ = 0.0
        self.threshold = threshold

        for _ in range(max_iter):
            for rec in train_records:
                z = sum(self.weights_[f] * float(rec[f]) for f in self.features) + self.bias_
                # Sigmoid with numerical clipping
                p = 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))
                y = float(rec[LABEL_FIELD])
                grad = p - y

                for f in self.features:
                    self.weights_[f] -= learning_rate * grad * float(rec[f])
                self.bias_ -= learning_rate * grad

        self.is_fitted = True
        return self

    def predict_proba(self, records: List[Dict[str, Any]]) -> List[float]:
        """Inference: predict positive class probabilities without training data or credentials."""
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted or loaded before calling predict_proba()")

        probas: List[float] = []
        for rec in records:
            validate_inference_record(rec, self.features)
            z = sum(self.weights_[f] * float(rec[f]) for f in self.features) + self.bias_
            p = 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))
            probas.append(p)
        return probas

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        """Inference: predict binary class labels without training data or credentials."""
        probas = self.predict_proba(records)
        return [1 if p >= self.threshold else 0 for p in probas]

    def to_dict(self) -> Dict[str, Any]:
        """Export model parameters and schema into a serializable dictionary."""
        if not self.is_fitted:
            raise RuntimeError("Cannot export an unfitted model")
        return {
            "format_version": self.format_version,
            "model_type": "logistic_regression",
            "schema": {
                "id_field": ID_FIELD,
                "features": list(self.features),
                "label_field": LABEL_FIELD,
            },
            "parameters": {
                "weights": dict(self.weights_),
                "bias": float(self.bias_),
                "threshold": float(self.threshold),
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Export model to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_model(self, filepath: str) -> None:
        """Save model to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SavedClassificationPipeline":
        """Reconstruct model from dictionary with strict version and schema validation."""
        if not isinstance(data, dict):
            raise SchemaValidationError(f"Expected dict, got {type(data).__name__}")

        version = data.get("format_version")
        if version not in SUPPORTED_FORMAT_VERSIONS:
            raise SchemaVersionMismatchError(
                f"Unsupported format_version '{version}'. Supported: {sorted(SUPPORTED_FORMAT_VERSIONS)}"
            )

        schema = data.get("schema")
        if not isinstance(schema, dict) or "features" not in schema:
            raise SchemaValidationError("Invalid or missing schema in model payload")

        features = schema["features"]
        if not isinstance(features, list) or len(features) == 0:
            raise SchemaValidationError("Schema 'features' must be a non-empty list")

        params = data.get("parameters")
        if not isinstance(params, dict) or "weights" not in params or "bias" not in params:
            raise SchemaValidationError("Invalid or missing parameters in model payload")

        weights = {f: float(params["weights"][f]) for f in features if f in params["weights"]}
        bias = float(params["bias"])
        threshold = float(params.get("threshold", 0.5))

        return cls(
            weights=weights,
            bias=bias,
            threshold=threshold,
            features=features,
            format_version=version,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "SavedClassificationPipeline":
        """Deserialize model from JSON string."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"Invalid JSON string: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def load_model(cls, filepath: str) -> "SavedClassificationPipeline":
        """Load model from JSON file."""
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        return cls.from_json(content)


def verify_predictions_agreement(
    probas_before: List[float],
    probas_after: List[float],
    preds_before: List[int],
    preds_after: List[int],
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """Verify that predictions before and after serialization agree within tolerance."""
    if len(probas_before) != len(probas_after) or len(preds_before) != len(preds_after):
        return False

    for pb, pa in zip(probas_before, probas_after):
        if abs(pb - pa) > tolerance:
            return False

    return preds_before == preds_after
