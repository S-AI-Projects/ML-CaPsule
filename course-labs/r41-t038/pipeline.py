"""Error analysis reporting pipeline for Mobile price range classification (Task R41-T038)."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ErrorRecord:
    record_id: str
    group_id: str
    expected_value: int
    predicted_value: int
    reason_category: str
    feature_summary: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_id": self.record_id,
            "group_id": self.group_id,
            "expected_value": self.expected_value,
            "predicted_value": self.predicted_value,
            "reason_category": self.reason_category,
            "feature_summary": self.feature_summary,
        }


@dataclass
class ErrorAnalysisReport:
    total_records: int
    correct_count: int
    error_count: int
    error_rate: float
    category_counts: Dict[str, int]
    error_records: List[ErrorRecord]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "correct_count": self.correct_count,
            "error_count": self.error_count,
            "error_rate": self.error_rate,
            "category_counts": self.category_counts,
            "error_records": [e.to_dict() for e in self.error_records],
        }


class ClassificationErrorAnalyzer:
    """Analyzes classification predictions and produces structured error reports."""

    REASON_CATEGORIES = {
        (0, 1): "false_positive_boundary_overshoot",
        (1, 0): "false_negative_underestimation",
    }

    def analyze(
        self,
        records: List[Dict[str, Any]],
        predictions: Dict[str, int],
    ) -> ErrorAnalysisReport:
        """
        Produces detailed error analysis report.
        Correct predictions are excluded from error_records.
        Aggregate counts are strictly asserted to match detailed error records.
        """
        error_records: List[ErrorRecord] = []
        category_counts: Dict[str, int] = {}
        correct_count = 0

        for r in records:
            rid = r["record_id"]
            expected = int(r["label"])
            predicted = int(predictions[rid])

            if expected == predicted:
                correct_count += 1
            else:
                cat = self.REASON_CATEGORIES.get(
                    (expected, predicted), "misclassification"
                )
                category_counts[cat] = category_counts.get(cat, 0) + 1
                error_records.append(
                    ErrorRecord(
                        record_id=rid,
                        group_id=r.get("group_id", "default"),
                        expected_value=expected,
                        predicted_value=predicted,
                        reason_category=cat,
                        feature_summary={"x1": float(r["x1"]), "x2": float(r["x2"])},
                    )
                )

        total = len(records)
        error_count = len(error_records)
        error_rate = error_count / total if total > 0 else 0.0

        return ErrorAnalysisReport(
            total_records=total,
            correct_count=correct_count,
            error_count=error_count,
            error_rate=error_rate,
            category_counts=category_counts,
            error_records=error_records,
        )


def build_testable_predictions_with_injected_error(
    records: List[Dict[str, Any]],
    injected_wrong_id: str = "c7",
) -> Dict[str, int]:
    """
    Builds predictions that are all correct EXCEPT for exactly one deliberately wrong prediction (c7).
    c7 has expected label 1, but will be predicted as 0 (false negative).
    """
    predictions = {}
    for r in records:
        rid = r["record_id"]
        true_label = int(r["label"])
        if rid == injected_wrong_id:
            # Deliberately invert label to test error reporting
            predictions[rid] = 1 - true_label
        else:
            predictions[rid] = true_label
    return predictions


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load JSON fixture."""
    if path is None:
        path = Path(__file__).parent / "fixture.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
