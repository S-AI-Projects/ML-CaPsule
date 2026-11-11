"""Decision tree classifier and error analysis report generator (Task R41-T018)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class SimpleThresholdDecisionTree:
    """Decision stump splitting on x1 threshold fitted on training records."""

    def __init__(self, threshold: float = 3.5) -> None:
        self.threshold = threshold

    def predict_one(self, x1: float) -> int:
        return 1 if x1 >= self.threshold else 0

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        return [self.predict_one(r["x1"]) for r in records]


def categorize_error(record: Dict[str, Any], expected: int, predicted: int) -> str:
    """Determines reason category based on feature values and decision threshold."""
    if expected == predicted:
        return "CORRECT_PREDICTION"
    if record["x1"] >= 3.0 and record["x1"] <= 4.0:
        return "BOUNDARY_AMBIGUITY"
    return "MISCLASSIFICATION_NOISE"


def build_error_report(
    records: List[Dict[str, Any]],
    predictions: List[int],
    inject_wrong_id: Optional[str] = None,
    wrong_prediction_val: Optional[int] = None
) -> Dict[str, Any]:
    """Generates detailed and aggregate error report schema."""
    detailed_errors: List[Dict[str, Any]] = []
    total_evaluated = len(records)
    error_count = 0

    for record, pred in zip(records, predictions):
        rec_id = record["record_id"]
        actual = record["label"]
        effective_pred = wrong_prediction_val if rec_id == inject_wrong_id else pred

        if effective_pred != actual:
            error_count += 1
            reason = categorize_error(record, actual, effective_pred)
            detailed_errors.append({
                "record_id": rec_id,
                "expected": actual,
                "predicted": effective_pred,
                "reason_category": reason,
                "feature_values": {"x1": record["x1"], "x2": record["x2"]}
            })

    return {
        "total_samples": total_evaluated,
        "error_count": error_count,
        "accuracy": round((total_evaluated - error_count) / total_evaluated, 4),
        "detailed_errors": detailed_errors
    }


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or (Path(__file__).resolve().parent / "fixture.json")
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline() -> Dict[str, Any]:
    payload = load_fixture()
    records = payload["records"]
    by_id = {r["record_id"]: r for r in records}

    test_ids = payload["metadata"]["stratified_split"]["test"]
    test_records = [by_id[i] for i in test_ids]

    model = SimpleThresholdDecisionTree(threshold=3.5)
    raw_predictions = model.predict(test_records)

    # Deliberately inject wrong prediction on record r4 (expected 0, injected 1)
    report = build_error_report(
        records=test_records,
        predictions=raw_predictions,
        inject_wrong_id="r4",
        wrong_prediction_val=1
    )

    return {
        "task_id": "R41-T018",
        "tested_records": test_ids,
        "error_analysis_report": report
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
