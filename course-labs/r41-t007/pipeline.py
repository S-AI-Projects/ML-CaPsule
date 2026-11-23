"""Serializable Configuration and Classification Pipeline for Task R41-T007.

Implements:
1. Strict, JSON-serializable PipelineConfig with schema version, seed, feature ordering,
   split IDs, and algorithm settings.
2. Rejection of unknown keys, contradictory parameters, and incompatible input shapes.
3. ConfigurableClassifier whose training and inference are 100% reproducible across
   configuration serialization and deserialization.
"""

import json
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

EXPECTED_CONFIG_KEYS: Set[str] = {
    "schema_version",
    "seed",
    "feature_ordering",
    "split_ids",
    "algorithm_settings",
}

EXPECTED_ALGO_KEYS: Set[str] = {
    "algorithm",
    "learning_rate",
    "max_iter",
    "threshold",
}

SUPPORTED_SCHEMA_VERSIONS: Set[str] = {"1.0.0"}
SUPPORTED_ALGORITHMS: Set[str] = {"logistic_regression", "linear_classifier"}
ID_FIELD: str = "record_id"
LABEL_FIELD: str = "label"


class ConfigurationError(ValueError):
    """Raised when configuration contains unknown or contradictory settings."""
    pass


class IncompatibleShapeError(ValueError):
    """Raised when record features do not match configured feature ordering."""
    pass


class PipelineConfig:
    """Serializable, validated pipeline configuration."""

    def __init__(
        self,
        schema_version: str,
        seed: int,
        feature_ordering: List[str],
        split_ids: Dict[str, List[str]],
        algorithm_settings: Dict[str, Any],
    ) -> None:
        self.schema_version = schema_version
        self.seed = seed
        self.feature_ordering = list(feature_ordering)
        self.split_ids = {k: list(v) for k, v in split_ids.items()}
        self.algorithm_settings = dict(algorithm_settings)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "feature_ordering": list(self.feature_ordering),
            "split_ids": {k: list(v) for k, v in self.split_ids.items()},
            "algorithm_settings": dict(self.algorithm_settings),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        allowed_record_ids: Optional[Set[str]] = None,
    ) -> "PipelineConfig":
        """Construct and validate PipelineConfig from dictionary."""
        if not isinstance(data, dict):
            raise ConfigurationError(f"Config must be a dict, got {type(data).__name__}")

        # 1. Reject unknown configuration keys
        unknown_keys = set(data.keys()) - EXPECTED_CONFIG_KEYS
        if unknown_keys:
            raise ConfigurationError(f"Unknown configuration key(s) detected: {sorted(unknown_keys)}")

        missing_keys = EXPECTED_CONFIG_KEYS - set(data.keys())
        if missing_keys:
            raise ConfigurationError(f"Missing required configuration key(s): {sorted(missing_keys)}")

        # 2. Schema version validation
        schema_version = data["schema_version"]
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ConfigurationError(
                f"Unsupported schema_version '{schema_version}'. Expected one of {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )

        # 3. Seed validation
        seed = data["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ConfigurationError(f"'seed' must be a non-negative integer, got {seed!r}")

        # 4. Feature ordering validation
        features = data["feature_ordering"]
        if not isinstance(features, list) or len(features) == 0:
            raise ConfigurationError("'feature_ordering' must be a non-empty list of feature names")
        if len(features) != len(set(features)):
            raise ConfigurationError(f"Duplicate features found in 'feature_ordering': {features}")
        for feat in features:
            if not isinstance(feat, str) or not feat.strip():
                raise ConfigurationError(f"Feature name must be a non-empty string, got {feat!r}")

        # 5. Split IDs validation
        split_ids = data["split_ids"]
        if not isinstance(split_ids, dict):
            raise ConfigurationError("'split_ids' must be a dictionary")
        required_splits = {"train", "validation", "test"}
        if set(split_ids.keys()) != required_splits:
            raise ConfigurationError(
                f"'split_ids' keys must exactly match {required_splits}, got {set(split_ids.keys())}"
            )

        seen_split_ids: Set[str] = set()
        for s_name, ids in split_ids.items():
            if not isinstance(ids, list) or len(ids) == 0:
                raise ConfigurationError(f"Split '{s_name}' must contain a non-empty list of IDs")
            for rid in ids:
                if not isinstance(rid, str) or not rid.strip():
                    raise ConfigurationError(f"Invalid record_id '{rid}' in split '{s_name}'")
                if rid in seen_split_ids:
                    raise ConfigurationError(
                        f"Contradictory split: Record ID '{rid}' appears in multiple split partitions"
                    )
                seen_split_ids.add(rid)
                if allowed_record_ids is not None and rid not in allowed_record_ids:
                    raise ConfigurationError(
                        f"Contradictory split: Record ID '{rid}' in split '{s_name}' does not exist in dataset"
                    )

        # 6. Algorithm settings validation
        algo_settings = data["algorithm_settings"]
        if not isinstance(algo_settings, dict):
            raise ConfigurationError("'algorithm_settings' must be a dictionary")

        unknown_algo_keys = set(algo_settings.keys()) - EXPECTED_ALGO_KEYS
        if unknown_algo_keys:
            raise ConfigurationError(f"Unknown algorithm setting(s) detected: {sorted(unknown_algo_keys)}")

        missing_algo_keys = EXPECTED_ALGO_KEYS - set(algo_settings.keys())
        if missing_algo_keys:
            raise ConfigurationError(f"Missing required algorithm setting(s): {sorted(missing_algo_keys)}")

        algo_name = algo_settings["algorithm"]
        if algo_name not in SUPPORTED_ALGORITHMS:
            raise ConfigurationError(
                f"Unsupported algorithm '{algo_name}'. Expected one of {sorted(SUPPORTED_ALGORITHMS)}"
            )

        lr = algo_settings["learning_rate"]
        if isinstance(lr, bool) or not isinstance(lr, (int, float)) or lr <= 0:
            raise ConfigurationError(f"'learning_rate' must be a positive number, got {lr!r}")

        max_iter = algo_settings["max_iter"]
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ConfigurationError(f"'max_iter' must be a positive integer >= 1, got {max_iter!r}")

        threshold = algo_settings["threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not (0.0 < threshold < 1.0):
            raise ConfigurationError(f"'threshold' must be a float between 0 and 1, got {threshold!r}")

        return cls(
            schema_version=schema_version,
            seed=seed,
            feature_ordering=features,
            split_ids=split_ids,
            algorithm_settings=algo_settings,
        )

    @classmethod
    def from_json(
        cls,
        json_str: str,
        allowed_record_ids: Optional[Set[str]] = None,
    ) -> "PipelineConfig":
        """Deserialize and validate PipelineConfig from a JSON string."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON string: {exc}") from exc
        return cls.from_dict(data, allowed_record_ids=allowed_record_ids)


class ConfigurableClassifier:
    """Classifier operating under a validated PipelineConfig."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.weights_: Dict[str, float] = {}
        self.bias_: float = 0.0
        self.is_fitted: bool = False

    def validate_record_shape(self, record: Dict[str, Any]) -> None:
        """Ensure record matches the exact configured feature ordering and shape."""
        if not isinstance(record, dict):
            raise IncompatibleShapeError("Record must be a dict")
        if ID_FIELD not in record:
            raise IncompatibleShapeError(f"Record missing '{ID_FIELD}'")

        # Every feature in feature_ordering must be present and numeric
        for feat in self.config.feature_ordering:
            if feat not in record:
                raise IncompatibleShapeError(
                    f"Incompatible shape: Record '{record.get(ID_FIELD)}' is missing configured feature '{feat}'"
                )
            val = record[feat]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise IncompatibleShapeError(
                    f"Incompatible shape: Feature '{feat}' must be numeric, got {type(val).__name__}"
                )

    def partition_records(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Partition records according to the configured split_ids."""
        records_by_id = {r[ID_FIELD]: r for r in records}
        partitioned: Dict[str, List[Dict[str, Any]]] = {}
        for split_name, ids in self.config.split_ids.items():
            split_rows: List[Dict[str, Any]] = []
            for rid in ids:
                if rid not in records_by_id:
                    raise ConfigurationError(f"Split ID '{rid}' not present in provided records")
                split_rows.append(records_by_id[rid])
            partitioned[split_name] = split_rows
        return partitioned

    def fit(self, train_records: List[Dict[str, Any]]) -> "ConfigurableClassifier":
        """Fit binary classifier using configured seed and hyperparameters."""
        for rec in train_records:
            self.validate_record_shape(rec)
            if LABEL_FIELD not in rec or rec[LABEL_FIELD] not in (0, 1):
                raise ValueError(f"Training record must have binary label in {{0, 1}}")

        rng = random.Random(self.config.seed)
        self.weights_ = {feat: rng.uniform(-0.05, 0.05) for feat in self.config.feature_ordering}
        self.bias_ = 0.0

        lr = float(self.config.algorithm_settings["learning_rate"])
        max_iter = int(self.config.algorithm_settings["max_iter"])

        for _ in range(max_iter):
            for rec in train_records:
                z = sum(self.weights_[f] * float(rec[f]) for f in self.config.feature_ordering) + self.bias_
                # Numerically stable sigmoid
                p = 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))
                y = float(rec[LABEL_FIELD])
                grad = p - y

                for f in self.config.feature_ordering:
                    self.weights_[f] -= lr * grad * float(rec[f])
                self.bias_ -= lr * grad

        self.is_fitted = True
        return self

    def predict_proba(self, records: List[Dict[str, Any]]) -> List[float]:
        """Predict positive class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict")
        probas: List[float] = []
        for rec in records:
            self.validate_record_shape(rec)
            z = sum(self.weights_[f] * float(rec[f]) for f in self.config.feature_ordering) + self.bias_
            p = 1.0 / (1.0 + math.exp(-max(min(z, 20.0), -20.0)))
            probas.append(p)
        return probas

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        """Predict binary class labels [0, 1]."""
        threshold = float(self.config.algorithm_settings["threshold"])
        probas = self.predict_proba(records)
        return [1 if p >= threshold else 0 for p in probas]
