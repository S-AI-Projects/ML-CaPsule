"""Input perturbation and robustness testing pipeline (Task R41-T019)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class RobustLinearRegressionInterface:
    """Regression interface supporting ID alignment and strict input validation."""

    def __init__(self, w1: float = 2.0, w2: float = 1.0) -> None:
        self.w1 = w1
        self.w2 = w2

    def validate_record(self, record: Dict[str, Any]) -> None:
        if "record_id" not in record:
            raise ValueError("Record missing required 'record_id'")
        for field in ("x1", "x2"):
            if field not in record:
                raise ValueError(f"Record {record.get('record_id')} missing field: {field}")
            val = record[field]
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"Record {record.get('record_id')} feature {field} must be numeric")

    def predict_one(self, record: Dict[str, Any]) -> float:
        self.validate_record(record)
        return round(self.w1 * float(record["x1"]) + self.w2 * float(record["x2"]), 4)

    def predict_records(self, records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Predicts and returns mapping keyed by stable record_id."""
        return {r["record_id"]: self.predict_one(r) for r in records}

    def predict_with_permutation_alignment(
        self, canonical_records: List[Dict[str, Any]], perturbed_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Verifies prediction invariance under input row order perturbation."""
        preds_canonical = self.predict_records(canonical_records)
        preds_perturbed = self.predict_records(perturbed_records)

        # ID-aligned comparison
        aligned_diffs = {
            rid: abs(preds_canonical[rid] - preds_perturbed[rid])
            for rid in preds_canonical
            if rid in preds_perturbed
        }
        max_diff = max(aligned_diffs.values()) if aligned_diffs else 0.0
        is_preserved = len(preds_canonical) == len(preds_perturbed) and max_diff < 1e-6

        return {
            "is_preserved": is_preserved,
            "max_id_aligned_diff": max_diff,
            "canonical_order": [r["record_id"] for r in canonical_records],
            "perturbed_order": [r["record_id"] for r in perturbed_records],
        }


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or (Path(__file__).resolve().parent / "fixture.json")
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline() -> Dict[str, Any]:
    payload = load_fixture()
    records = payload["records"]

    # Valid perturbation: reverse record order while keeping stable IDs
    perturbed_records = list(reversed(records))

    interface = RobustLinearRegressionInterface(w1=2.0, w2=1.0)
    perturbation_eval = interface.predict_with_permutation_alignment(records, perturbed_records)
    predictions = interface.predict_records(records)

    return {
        "task_id": "R41-T019",
        "sample_count": len(records),
        "predictions": predictions,
        "perturbation_assessment": perturbation_eval,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
