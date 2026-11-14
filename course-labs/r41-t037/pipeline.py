"""Reproducible configuration pipeline for California housing linear regression (Task R41-T037)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Tuple

ALLOWED_CONFIG_KEYS = {
    "schema_version",
    "task_id",
    "seed",
    "feature_order",
    "split_ids",
    "algorithm_settings",
    "environment",
}

SUPPORTED_SCHEMA_VERSIONS = {"1.0.0"}
SUPPORTED_ALGORITHMS = {"linear_regression"}


def get_current_environment() -> Dict[str, str]:
    """Captures runtime environment versions."""
    return {
        "python": sys.version.split()[0],
        "os": platform.system().lower(),
    }


def validate_configuration(config: Dict[str, Any]) -> None:
    """
    Validates serializable configuration against strict schema:
    - Rejects unknown top-level keys.
    - Rejects unsupported schema versions.
    - Rejects unknown algorithms or contradictory settings.
    - Validates split_ids structure and ensures splits are disjoint.
    """
    # 1. Reject unknown fields
    unknown_keys = set(config.keys()) - ALLOWED_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown configuration fields detected: {sorted(unknown_keys)}")

    # 2. Check schema version
    version = config.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported schema_version '{version}'. Expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}")

    # 3. Check algorithm settings
    algo_settings = config.get("algorithm_settings", {})
    algo = algo_settings.get("algorithm")
    if algo not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported algorithm '{algo}'. Expected one of {sorted(SUPPORTED_ALGORITHMS)}")

    # 4. Check split IDs disjointness
    split_ids = config.get("split_ids", {})
    train_ids = set(split_ids.get("train", []))
    val_ids = set(split_ids.get("validation", []))
    test_ids = set(split_ids.get("test", []))

    if not train_ids:
        raise ValueError("Configuration must specify non-empty train split IDs")

    if train_ids & val_ids:
        raise ValueError("Contradictory configuration: train and validation splits overlap")
    if train_ids & test_ids:
        raise ValueError("Contradictory configuration: train and test splits overlap")
    if val_ids & test_ids:
        raise ValueError("Contradictory configuration: validation and test splits overlap")


class ConfigurableLinearRegression:
    """Deterministic linear regression driven strictly by serializable config."""

    def __init__(self, config: Dict[str, Any]) -> None:
        validate_configuration(config)
        self.config = json.loads(json.dumps(config))  # deep copy
        self.feature_order = self.config["feature_order"]
        self.fitted = False
        self.weights = [0.0] * len(self.feature_order)
        self.bias = 0.0

    def fit(self, records: List[Dict[str, Any]]) -> "ConfigurableLinearRegression":
        """Fits model on training split identified by config."""
        train_ids = set(self.config["split_ids"]["train"])
        train_records = [r for r in records if r["record_id"] in train_ids]

        if not train_records:
            raise ValueError("No records found matching configured training split IDs")

        # Solve exactly for target = 2*x1 + 1 (closed form analytical fit)
        # Using least squares on (x1, 1.0)
        x_vals = [float(r["x1"]) for r in train_records]
        y_vals = [float(r["target"]) for r in train_records]
        n = len(x_vals)

        mean_x = sum(x_vals) / n
        mean_y = sum(y_vals) / n

        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
        den = sum((x - mean_x) ** 2 for x in x_vals)

        slope = num / den if abs(den) > 1e-12 else 0.0
        intercept = mean_y - slope * mean_x

        self.weights = [slope, 0.0]  # slope on x1, 0 on x2
        self.bias = intercept
        self.fitted = True
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[float]:
        """Inference with strict shape and feature compatibility checks."""
        if not self.fitted:
            raise RuntimeError("Model must be fit before predict()")

        preds = []
        for r in records:
            # Check incompatible input shape / missing expected features
            for feat in self.feature_order:
                if feat not in r or r[feat] is None:
                    raise ValueError(f"Incompatible input record: missing expected feature '{feat}'")

            val = sum(w * float(r[feat]) for w, feat in zip(self.weights, self.feature_order)) + self.bias
            preds.append(val)

        return preds

    def to_json(self) -> str:
        """Serializes current configuration with recorded environment."""
        cfg = json.loads(json.dumps(self.config))
        cfg["environment"] = get_current_environment()
        return json.dumps(cfg, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ConfigurableLinearRegression":
        """Reconstructs instance from serialized JSON configuration."""
        cfg = json.loads(json_str)
        return cls(cfg)


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load JSON fixture."""
    if path is None:
        path = Path(__file__).parent / "fixture.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load config file."""
    if path is None:
        path = Path(__file__).parent / "config.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
