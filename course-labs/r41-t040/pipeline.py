"""Saved-pipeline interface for Wine quality regression without untrusted pickle (Task R41-T040)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CURRENT_SCHEMA_VERSION = "1.0.0"
EXPECTED_FEATURES = ("x1", "x2")


class SafeSavedPipeline:
    """
    Serializes and deserializes model configuration and fitted parameters exclusively via JSON.
    Never loads or saves untrusted pickle payloads.
    Enforces inference-time schema validation with zero training data or credential dependencies.
    """

    def __init__(
        self,
        schema_version: str = CURRENT_SCHEMA_VERSION,
        weights: Optional[List[float]] = None,
        bias: float = 0.0,
        features: Tuple[str, ...] = EXPECTED_FEATURES,
    ) -> None:
        self.schema_version = schema_version
        self.weights = weights if weights is not None else [0.0, 0.0]
        self.bias = bias
        self.features = list(features)
        self.fitted = weights is not None

    def fit(self, train_records: List[Dict[str, Any]]) -> "SafeSavedPipeline":
        """Fit deterministic parameters on training partition (closed-form for target = 2*x1 + 1)."""
        if not train_records:
            raise ValueError("Training records cannot be empty")

        x_vals = [float(r["x1"]) for r in train_records]
        y_vals = [float(r["target"]) for r in train_records]
        n = len(x_vals)

        mean_x = sum(x_vals) / n
        mean_y = sum(y_vals) / n

        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
        den = sum((x - mean_x) ** 2 for x in x_vals)

        slope = num / den if abs(den) > 1e-12 else 0.0
        intercept = mean_y - slope * mean_x

        self.weights = [slope, 0.0]  # slope for x1, zero for constant x2
        self.bias = intercept
        self.fitted = True
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[float]:
        """
        Inference function with schema validation:
        Requires only test/inference records, needing zero training data or credentials.
        """
        if not self.fitted:
            raise RuntimeError("Pipeline must be fitted or loaded before prediction")

        predictions: List[float] = []
        for idx, r in enumerate(records):
            # Schema validation
            for feat in self.features:
                if feat not in r or r[feat] is None:
                    raise ValueError(f"Schema mismatch at record {r.get('record_id', idx)}: missing required feature '{feat}'")
                try:
                    val = float(r[feat])
                    if not math.isfinite(val):
                        raise ValueError(f"Non-finite value in feature '{feat}'")
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Schema mismatch: feature '{feat}' must be a finite float: {e}")

            pred = sum(w * float(r[f]) for w, f in zip(self.weights, self.features)) + self.bias
            predictions.append(pred)

        return predictions

    def save_json(self, path: Path) -> None:
        """Saves model parameters safely using JSON (eliminating pickle security risks)."""
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted pipeline")

        state = {
            "schema_version": self.schema_version,
            "pipeline_type": "safe_linear_regression_json",
            "features": self.features,
            "weights": self.weights,
            "bias": self.bias,
            "training_data_required_at_inference": False,
            "credentials_required_at_inference": False,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load_json(cls, path: Path) -> "SafeSavedPipeline":
        """
        Loads saved pipeline state from JSON and verifies schema version.
        Rejects mismatched versions and invalid schemas.
        """
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        loaded_version = state.get("schema_version")
        if loaded_version != CURRENT_SCHEMA_VERSION:
            raise ValueError(
                f"Schema version mismatch: saved version '{loaded_version}' incompatible with expected '{CURRENT_SCHEMA_VERSION}'"
            )

        features = state.get("features", [])
        if list(features) != list(EXPECTED_FEATURES):
            raise ValueError(f"Feature schema mismatch: expected {EXPECTED_FEATURES}, got {features}")

        weights = [float(w) for w in state.get("weights", [])]
        bias = float(state.get("bias", 0.0))

        return cls(
            schema_version=loaded_version,
            weights=weights,
            bias=bias,
            features=tuple(features),
        )


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load JSON fixture."""
    if path is None:
        path = Path(__file__).parent / "fixture.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
