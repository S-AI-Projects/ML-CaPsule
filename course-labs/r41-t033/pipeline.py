"""Minimal training-mean baseline for Palmer penguins regression fixture (Task R41-T033)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BaselineEvaluation:
    split: str
    record_ids: List[str]
    actual: List[float]
    predicted: List[float]
    mae: float


class TrainingMeanBaseline:
    """Simplest honest regression baseline: predicts the mean of the training targets."""

    def __init__(self) -> None:
        self.training_mean: Optional[float] = None
        self.fitted: bool = False
        self.train_record_ids: List[str] = []

    def fit(self, train_records: List[Dict[str, Any]]) -> "TrainingMeanBaseline":
        """Fits the baseline using ONLY training records."""
        if not train_records:
            raise ValueError("Cannot fit baseline on empty training records")

        targets = []
        for r in train_records:
            if "target" not in r or r["target"] is None:
                raise ValueError(f"Record {r.get('record_id')} is missing 'target'")
            val = float(r["target"])
            if not math.isfinite(val):
                raise ValueError(f"Non-finite target in record {r.get('record_id')}")
            targets.append(val)

        self.training_mean = sum(targets) / len(targets)
        self.train_record_ids = [r["record_id"] for r in train_records]
        self.fitted = True
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[float]:
        """Predicts the learned training mean for all incoming records."""
        if not self.fitted or self.training_mean is None:
            raise RuntimeError("Baseline model must be fit before calling predict()")
        return [self.training_mean for _ in records]

    def evaluate(self, records: List[Dict[str, Any]], split_name: str = "held-out") -> BaselineEvaluation:
        """Evaluates baseline MAE on given held-out records."""
        preds = self.predict(records)
        actuals = [float(r["target"]) for r in records]
        ids = [r["record_id"] for r in records]

        abs_errors = [abs(a - p) for a, p in zip(actuals, preds)]
        mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0.0

        return BaselineEvaluation(
            split=split_name,
            record_ids=ids,
            actual=actuals,
            predicted=preds,
            mae=mae,
        )


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load JSON fixture."""
    if path is None:
        path = Path(__file__).parent / "fixture.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
