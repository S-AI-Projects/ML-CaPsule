"""Bounded CPU smoke run pipeline for FIFA players rating regression (Task R41-T036)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SmokeRunSummary:
    """Records the execution metrics of the bounded CPU smoke test."""

    sample_count: int
    step_count: int
    final_loss: float
    is_finite: bool
    prediction_shape: Tuple[int, int]
    hardware_used: str
    model_type: str


class CPUSmokeRegressionPipeline:
    """
    Tiny CPU-only regression smoke test bounded to <=32 samples and <=20 update steps.
    Asserts interface correctness and finite outputs without claiming production accuracy.
    """

    def __init__(self, seed: int = 42, max_steps: int = 20) -> None:
        self.seed = seed
        self.max_steps = min(max_steps, 20)
        self.fitted = False
        self.step_count = 0
        self.weights = [0.0, 0.0]  # x1, x2
        self.bias = 0.0
        self.last_loss = float("nan")

    def fit_smoke_step(self, train_records: List[Dict[str, Any]]) -> SmokeRunSummary:
        """
        Executes bounded gradient steps on CPU for at most 20 steps.
        Enforces sample count <= 32.
        """
        if len(train_records) > 32:
            raise ValueError(f"Smoke test exceeded sample bound: {len(train_records)} > 32")
        if not train_records:
            raise ValueError("Training records cannot be empty")

        learning_rate = 0.01
        self.step_count = 0

        # Execute bounded CPU gradient updates
        for step in range(self.max_steps):
            total_loss = 0.0
            grad_w0 = 0.0
            grad_w1 = 0.0
            grad_b = 0.0
            n = len(train_records)

            for r in train_records:
                x1 = float(r["x1"])
                x2 = float(r["x2"])
                y = float(r["target"])

                y_pred = self.weights[0] * x1 + self.weights[1] * x2 + self.bias
                err = y_pred - y
                total_loss += err * err

                grad_w0 += (2.0 / n) * err * x1
                grad_w1 += (2.0 / n) * err * x2
                grad_b += (2.0 / n) * err

            self.weights[0] -= learning_rate * grad_w0
            self.weights[1] -= learning_rate * grad_w1
            self.bias -= learning_rate * grad_b

            self.step_count += 1
            self.last_loss = total_loss / n

        self.fitted = True

        return SmokeRunSummary(
            sample_count=len(train_records),
            step_count=self.step_count,
            final_loss=self.last_loss,
            is_finite=math.isfinite(self.last_loss),
            prediction_shape=(len(train_records), 1),
            hardware_used="CPU",
            model_type="deterministic_cpu_linear_sgd_baseline",
        )

    def predict(self, records: List[Dict[str, Any]]) -> List[float]:
        """Inference with strict schema validation."""
        if not self.fitted:
            raise RuntimeError("Model must be trained with fit_smoke_step() before predict()")

        preds = []
        for r in records:
            if "x1" not in r or "x2" not in r:
                raise ValueError("Record missing required features 'x1' or 'x2'")
            x1 = float(r["x1"])
            x2 = float(r["x2"])
            y_pred = self.weights[0] * x1 + self.weights[1] * x2 + self.bias
            if not math.isfinite(y_pred):
                raise ValueError("Prediction resulted in non-finite value")
            preds.append(y_pred)

        return preds


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load JSON fixture."""
    if path is None:
        path = Path(__file__).parent / "fixture.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
