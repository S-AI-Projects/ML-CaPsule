"""Bounded CPU Smoke Run Pipeline for Task R41-T006: Forest fires – regression.

Implements a bounded, offline, CPU-only regression smoke test:
1. Maximum 32 samples boundary check.
2. Fixed random seed for deterministic execution.
3. Bounded training: at most 1 epoch and at most 20 update steps.
4. Linear SGD regression model and an explicitly labelled constant/naive baseline.
5. Verifies finite outputs, finite loss, and expected prediction shape.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

FEATURE_ORDER: List[str] = ["x1", "x2"]
TARGET_FIELD: str = "target"
ID_FIELD: str = "record_id"
MAX_ALLOWED_SAMPLES: int = 32
MAX_ALLOWED_EPOCHS: int = 1
MAX_ALLOWED_STEPS: int = 20


class PipelineBoundaryError(ValueError):
    """Raised when bounds on sample count, epochs, or steps are violated."""
    pass


class SchemaValidationError(TypeError):
    """Raised when record types or schemas violate the contract."""
    pass


def validate_records(
    records: List[Dict[str, Any]],
    require_target: bool = False,
    max_samples: int = MAX_ALLOWED_SAMPLES,
) -> None:
    """Validate records, ensuring scalar numerics and sample count bound."""
    if not isinstance(records, list) or len(records) == 0:
        raise SchemaValidationError("Records must be a non-empty list of dicts")

    if len(records) > max_samples:
        raise PipelineBoundaryError(
            f"Sample count {len(records)} exceeds maximum allowed limit of {max_samples}"
        )

    seen_ids = set()
    for rec in records:
        if not isinstance(rec, dict):
            raise SchemaValidationError("Each record must be a dict")
        if ID_FIELD not in rec:
            raise SchemaValidationError(f"Missing required field '{ID_FIELD}'")
        rid = rec[ID_FIELD]
        if not isinstance(rid, str) or not rid.strip():
            raise SchemaValidationError(f"'{ID_FIELD}' must be a non-empty string")
        if rid in seen_ids:
            raise ValueError(f"Duplicate record_id '{rid}'")
        seen_ids.add(rid)

        for feat in FEATURE_ORDER:
            if feat not in rec:
                raise SchemaValidationError(f"Missing feature '{feat}' in record '{rid}'")
            val = rec[feat]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise SchemaValidationError(f"Feature '{feat}' must be numeric, got {type(val).__name__}")
            if math.isnan(val) or math.isinf(val):
                raise SchemaValidationError(f"Feature '{feat}' in record '{rid}' must be finite")

        if require_target:
            if TARGET_FIELD not in rec:
                raise SchemaValidationError(f"Missing target field '{TARGET_FIELD}' in record '{rid}'")
            t_val = rec[TARGET_FIELD]
            if isinstance(t_val, bool) or not isinstance(t_val, (int, float)):
                raise SchemaValidationError(f"Target must be numeric, got {type(t_val).__name__}")
            if math.isnan(t_val) or math.isinf(t_val):
                raise SchemaValidationError(f"Target in record '{rid}' must be finite")


class ConstantBaselineModel:
    """Explicitly labelled naive/constant baseline predicting the mean training target."""

    def __init__(self) -> None:
        self.model_type: str = "constant_mean_baseline"
        self.is_naive: bool = True
        self.mean_target_: Optional[float] = None
        self.step_count: int = 0
        self.train_loss: Optional[float] = None

    def fit(self, records: List[Dict[str, Any]]) -> "ConstantBaselineModel":
        validate_records(records, require_target=True)
        targets = [float(r[TARGET_FIELD]) for r in records]
        self.mean_target_ = sum(targets) / len(targets)
        self.step_count = 1  # 1 aggregation step
        self.train_loss = sum((t - self.mean_target_) ** 2 for t in targets) / len(targets)
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[float]:
        if self.mean_target_ is None:
            raise RuntimeError("Model must be fitted before predict")
        validate_records(records, require_target=False)
        return [float(self.mean_target_) for _ in records]


class CPURegressionSmokeModel:
    """Lightweight CPU-only linear regression trained with bounded SGD."""

    def __init__(
        self,
        learning_rate: float = 0.05,
        seed: int = 42,
        max_epochs: int = MAX_ALLOWED_EPOCHS,
        max_steps: int = MAX_ALLOWED_STEPS,
    ) -> None:
        if max_epochs > MAX_ALLOWED_EPOCHS:
            raise PipelineBoundaryError(f"max_epochs cannot exceed {MAX_ALLOWED_EPOCHS}")
        if max_steps > MAX_ALLOWED_STEPS:
            raise PipelineBoundaryError(f"max_steps cannot exceed {MAX_ALLOWED_STEPS}")

        self.learning_rate: float = learning_rate
        self.seed: int = seed
        self.max_epochs: int = max_epochs
        self.max_steps: int = max_steps
        self.model_type: str = "linear_sgd"
        self.is_naive: bool = False

        self.weights_: Dict[str, float] = {}
        self.bias_: float = 0.0
        self.step_count: int = 0
        self.epoch_count: int = 0
        self.train_loss: Optional[float] = None
        self.is_fitted: bool = False

    def fit(self, records: List[Dict[str, Any]]) -> "CPURegressionSmokeModel":
        validate_records(records, require_target=True, max_samples=MAX_ALLOWED_SAMPLES)

        # Deterministic weight initialization using fixed seed
        rng = random.Random(self.seed)
        self.weights_ = {feat: rng.uniform(-0.1, 0.1) for feat in FEATURE_ORDER}
        self.bias_ = 0.0
        self.step_count = 0
        self.epoch_count = 0

        # Execute bounded training: at most 1 epoch and at most max_steps updates
        for epoch in range(self.max_epochs):
            self.epoch_count += 1
            for rec in records:
                if self.step_count >= self.max_steps:
                    break

                # Forward pass: y_hat = w1*x1 + w2*x2 + b
                y_hat = sum(self.weights_[feat] * float(rec[feat]) for feat in FEATURE_ORDER) + self.bias_
                y_true = float(rec[TARGET_FIELD])
                error = y_hat - y_true  # residual

                # SGD update step: w = w - lr * error * x, b = b - lr * error
                for feat in FEATURE_ORDER:
                    grad_w = error * float(rec[feat])
                    self.weights_[feat] -= self.learning_rate * grad_w
                self.bias_ -= self.learning_rate * error

                self.step_count += 1

        self.is_fitted = True

        # Compute post-training loss on training set
        preds = self.predict(records)
        targets = [float(r[TARGET_FIELD]) for r in records]
        self.train_loss = sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(targets)
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling predict()")
        validate_records(records, require_target=False, max_samples=MAX_ALLOWED_SAMPLES)

        predictions: List[float] = []
        for rec in records:
            y_hat = sum(self.weights_[feat] * float(rec[feat]) for feat in FEATURE_ORDER) + self.bias_
            if math.isnan(y_hat) or math.isinf(y_hat):
                raise ValueError("Prediction produced non-finite output")
            predictions.append(float(y_hat))

        return predictions

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute MSE and MAE on evaluated records."""
        validate_records(records, require_target=True, max_samples=MAX_ALLOWED_SAMPLES)
        preds = self.predict(records)
        targets = [float(r[TARGET_FIELD]) for r in records]

        mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / len(targets)
        mae = sum(abs(p - t) for p, t in zip(preds, targets)) / len(targets)

        if math.isnan(mse) or math.isinf(mse) or math.isnan(mae) or math.isinf(mae):
            raise ValueError("Loss evaluation produced non-finite value")

        return {"mse": float(mse), "mae": float(mae)}


def run_regression_smoke_pipeline(
    train_records: List[Dict[str, Any]],
    val_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute complete bounded CPU regression smoke run."""
    model = CPURegressionSmokeModel(seed=seed, max_epochs=1, max_steps=20)
    model.fit(train_records)

    val_preds = model.predict(val_records)
    val_metrics = model.evaluate(val_records)

    test_preds = model.predict(test_records)
    test_metrics = model.evaluate(test_records)

    # Constant baseline for benchmark comparison
    baseline = ConstantBaselineModel()
    baseline.fit(train_records)
    baseline_val_preds = baseline.predict(val_records)

    return {
        "model_type": model.model_type,
        "is_naive": model.is_naive,
        "seed": seed,
        "step_count": model.step_count,
        "epoch_count": model.epoch_count,
        "train_loss_mse": model.train_loss,
        "val_prediction_shape": (len(val_preds),),
        "val_predictions": val_preds,
        "val_mse": val_metrics["mse"],
        "val_mae": val_metrics["mae"],
        "test_prediction_shape": (len(test_preds),),
        "test_predictions": test_preds,
        "test_mse": test_metrics["mse"],
        "test_mae": test_metrics["mae"],
        "baseline_val_predictions": baseline_val_preds,
        "baseline_mean_target": baseline.mean_target_,
        "all_outputs_finite": all(
            not math.isnan(x) and not math.isinf(x)
            for x in val_preds + test_preds + [model.train_loss, val_metrics["mse"], test_metrics["mse"]]
        ),
    }
