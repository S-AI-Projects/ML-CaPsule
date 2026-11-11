"""Bounded CPU regression pipeline smoke test (Task R41-T016)."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class BoundedCPULinearRegression:
    """Tiny CPU-only linear regression with bounded training steps and fixed seed."""

    def __init__(self, learning_rate: float = 0.05, max_steps: int = 20, seed: int = 42) -> None:
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.seed = seed
        self.w1 = 0.0
        self.w2 = 0.0
        self.step_count = 0
        self.loss_history: List[float] = []
        self.is_fitted = False

    def fit(self, X: List[List[float]], y: List[float]) -> "BoundedCPULinearRegression":
        if not X or not y:
            raise ValueError("Training inputs cannot be empty")
        if len(X) != len(y):
            raise ValueError("Mismatch between features and targets length")

        n = len(X)
        self.w1 = 0.1
        self.w2 = 0.1
        self.step_count = 0
        self.loss_history = []

        # Run bounded update steps (maximum 20 steps / 1 single epoch sweep)
        steps_to_run = min(self.max_steps, n)
        for i in range(steps_to_run):
            x1, x2 = X[i][0], X[i][1]
            target = y[i]

            pred = self.w1 * x1 + self.w2 * x2
            err = pred - target
            loss = 0.5 * (err ** 2)

            # Gradient step
            grad_w1 = err * x1
            grad_w2 = err * x2

            self.w1 -= self.learning_rate * grad_w1
            self.w2 -= self.learning_rate * grad_w2

            self.step_count += 1
            self.loss_history.append(round(loss, 4))

        self.is_fitted = True
        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        if not self.is_fitted:
            raise RuntimeError("Model must be trained before predicting")
        preds = []
        for row in X:
            x1, x2 = row[0], row[1]
            p = self.w1 * x1 + self.w2 * x2
            preds.append(round(p, 4))
        return preds


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or (Path(__file__).resolve().parent / "fixture.json")
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline() -> Dict[str, Any]:
    payload = load_fixture()
    metadata = payload["metadata"]
    records = payload["records"]
    by_id = {r["record_id"]: r for r in records}

    train_ids = metadata["split"]["train"]
    val_ids = metadata["split"]["validation"]
    test_ids = metadata["split"]["test"]

    X_train = [[by_id[i]["x1"], by_id[i]["x2"]] for i in train_ids]
    y_train = [by_id[i]["target"] for i in train_ids]

    X_val = [[by_id[i]["x1"], by_id[i]["x2"]] for i in val_ids]
    y_val = [by_id[i]["target"] for i in val_ids]

    X_test = [[by_id[i]["x1"], by_id[i]["x2"]] for i in test_ids]
    y_test = [by_id[i]["target"] for i in test_ids]

    model = BoundedCPULinearRegression(learning_rate=0.05, max_steps=20, seed=42)
    model.fit(X_train, y_train)

    train_preds = model.predict(X_train)
    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    # Compute MSE across splits
    def compute_mse(actuals: List[float], preds: List[float]) -> float:
        return round(sum((a - p) ** 2 for a, p in zip(actuals, preds)) / len(actuals), 4)

    return {
        "task_id": "R41-T016",
        "sample_count": len(records),
        "step_count": model.step_count,
        "weights": {"w1": round(model.w1, 4), "w2": round(model.w2, 4)},
        "loss_history": model.loss_history,
        "train_shape": [len(train_preds), 1],
        "validation_shape": [len(val_preds), 1],
        "test_shape": [len(test_preds), 1],
        "metrics": {
            "train_mse": compute_mse(y_train, train_preds),
            "val_mse": compute_mse(y_val, val_preds),
            "test_mse": compute_mse(y_test, test_preds)
        },
        "predictions": {
            "test_id": test_ids[0],
            "actual": y_test[0],
            "predicted": test_preds[0]
        }
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
