"""Training-mean baseline and evaluation pipeline for Auto MPG regression (Task R41-T023)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class BaselineEvaluation:
    """Evaluation summary on a designated split."""

    split: str
    record_ids: List[str]
    actuals: List[float]
    predictions: List[float]
    mae: float
    mse: float
    rmse: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "record_ids": list(self.record_ids),
            "actuals": list(self.actuals),
            "predictions": list(self.predictions),
            "mae": self.mae,
            "mse": self.mse,
            "rmse": self.rmse,
        }


def get_environment_info() -> Dict[str, str]:
    """Capture runtime environment versions."""
    return {
        "python": sys.version.split()[0],
        "os": platform.system().lower(),
    }


def verify_fixture_formula(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify that fixture records strictly satisfy target = 2 * x1 + 1."""
    discrepancies = []
    for r in records:
        x1 = float(r["x1"])
        target = float(r["target"])
        expected = 2.0 * x1 + 1.0
        if abs(target - expected) > 1e-9:
            discrepancies.append({
                "record_id": r.get("record_id"),
                "x1": x1,
                "target": target,
                "expected": expected,
            })
    return {
        "formula_satisfied": len(discrepancies) == 0,
        "total_records": len(records),
        "discrepancies": discrepancies,
    }


class TrainingMeanBaseline:
    """
    The simplest honest regression baseline:
    Learns the scalar target mean exclusively from training records (r1..r4)
    and predicts that constant value across all queries.
    """

    def __init__(self) -> None:
        self.training_mean: Optional[float] = None
        self.train_record_ids: List[str] = []
        self.fitted: bool = False

    def fit(self, train_records: List[Dict[str, Any]]) -> "TrainingMeanBaseline":
        """Fit baseline exclusively on training partition records."""
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
        """Predict the learned training mean for all incoming records."""
        if not self.fitted or self.training_mean is None:
            raise RuntimeError("Baseline model must be fit before calling predict()")
        return [self.training_mean for _ in records]

    def evaluate(self, records: List[Dict[str, Any]], split_name: str = "held-out") -> BaselineEvaluation:
        """Evaluate baseline predictions against held-out ground truth targets."""
        if not records:
            raise ValueError(f"Cannot evaluate on empty records for split '{split_name}'")

        preds = self.predict(records)
        actuals = [float(r["target"]) for r in records]
        ids = [r["record_id"] for r in records]

        abs_errors = [abs(a - p) for a, p in zip(actuals, preds)]
        sq_errors = [(a - p) ** 2 for a, p in zip(actuals, preds)]

        mae = sum(abs_errors) / len(abs_errors)
        mse = sum(sq_errors) / len(sq_errors)
        rmse = math.sqrt(mse)

        return BaselineEvaluation(
            split=split_name,
            record_ids=ids,
            actuals=actuals,
            predictions=preds,
            mae=mae,
            mse=mse,
            rmse=rmse,
        )


class LinearRegressionComparator:
    """Linear regression model for comparison against the honest baseline."""

    def __init__(self) -> None:
        self.slope: Optional[float] = None
        self.intercept: Optional[float] = None

    def fit(self, train_records: List[Dict[str, Any]]) -> "LinearRegressionComparator":
        xs = [float(r["x1"]) for r in train_records]
        ys = [float(r["target"]) for r in train_records]
        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        ss_xx = sum((x - mean_x) ** 2 for x in xs)
        self.slope = ss_xy / ss_xx
        self.intercept = mean_y - self.slope * mean_x
        return self

    def predict(self, records: List[Dict[str, Any]]) -> List[float]:
        return [self.slope * float(r["x1"]) + self.intercept for r in records]

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        preds = self.predict(records)
        actuals = [float(r["target"]) for r in records]
        mse = sum((p - a) ** 2 for p, a in zip(preds, actuals)) / len(records)
        mae = sum(abs(p - a) for p, a in zip(preds, actuals)) / len(records)
        return {"mae": mae, "mse": mse}


def load_fixture(fixture_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load JSON fixture from disk."""
    if fixture_path is None:
        fixture_path = Path(__file__).parent / "fixture.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)
