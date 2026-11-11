"""Saved pipeline interface with schema validation for Naive Bayes (Task R41-T020)."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


class SafeNaiveBayesModel:
    """Safe Naive Bayes pipeline with JSON-serialized state and schema validation."""

    def __init__(self) -> None:
        self.schema_version: str = "1.0.0"
        self.feature_order: List[str] = ["x1", "x2"]
        self.classes: List[int] = [0, 1]
        self.class_priors: Dict[int, float] = {}
        self.feature_means: Dict[int, Dict[str, float]] = {}
        self.feature_vars: Dict[int, Dict[str, float]] = {}
        self.is_fitted: bool = False

    def fit(self, records: List[Dict[str, Any]]) -> None:
        total = len(records)
        by_class: Dict[int, List[Dict[str, Any]]] = {0: [], 1: []}
        for r in records:
            by_class[int(r["label"])].append(r)

        for c in self.classes:
            self.class_priors[c] = len(by_class[c]) / total
            self.feature_means[c] = {}
            self.feature_vars[c] = {}
            for feat in self.feature_order:
                vals = [float(r[feat]) for r in by_class[c]]
                mean_val = sum(vals) / len(vals)
                var_val = sum((v - mean_val) ** 2 for v in vals) / len(vals)
                self.feature_means[c][feat] = mean_val
                # Regularize variance to prevent division by zero
                self.feature_vars[c][feat] = max(var_val, 1e-4)

        self.is_fitted = True

    def validate_inference_record(self, record: Dict[str, Any]) -> None:
        if "record_id" not in record:
            raise ValueError("Inference input missing required field 'record_id'")
        for feat in self.feature_order:
            if feat not in record:
                raise ValueError(f"Inference record {record.get('record_id')} missing feature '{feat}'")
            val = record[feat]
            if not isinstance(val, (int, float)) or isinstance(val, bool):
                raise TypeError(f"Feature '{feat}' must be numeric, got {type(val).__name__}")

    def predict_log_likelihood(self, record: Dict[str, Any], class_label: int) -> float:
        log_prob = math.log(self.class_priors[class_label])
        for feat in self.feature_order:
            x = float(record[feat])
            mean = self.feature_means[class_label][feat]
            var = self.feature_vars[class_label][feat]
            log_prob += -0.5 * math.log(2 * math.pi * var) - ((x - mean) ** 2) / (2 * var)
        return log_prob

    def predict_one(self, record: Dict[str, Any]) -> int:
        if not self.is_fitted:
            raise RuntimeError("Model is not loaded or fitted")
        self.validate_inference_record(record)
        scores = {c: self.predict_log_likelihood(record, c) for c in self.classes}
        return max(scores, key=scores.get)

    def predict(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        return {r["record_id"]: self.predict_one(r) for r in records}

    def export_state_dict(self) -> Dict[str, Any]:
        if not self.is_fitted:
            raise RuntimeError("Cannot export unfitted model state")
        return {
            "model_type": "SafeNaiveBayesModel",
            "schema_version": self.schema_version,
            "feature_order": self.feature_order,
            "classes": self.classes,
            "class_priors": {str(k): v for k, v in self.class_priors.items()},
            "feature_means": {str(k): v for k, v in self.feature_means.items()},
            "feature_vars": {str(k): v for k, v in self.feature_vars.items()}
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("schema_version") != self.schema_version:
            raise ValueError(f"Schema version mismatch: expected {self.schema_version}, got {state.get('schema_version')}")
        if state.get("feature_order") != self.feature_order:
            raise ValueError(f"Feature order mismatch: expected {self.feature_order}, got {state.get('feature_order')}")

        self.classes = [int(c) for c in state["classes"]]
        self.class_priors = {int(k): float(v) for k, v in state["class_priors"].items()}
        self.feature_means = {int(k): {f: float(v) for f, v in fmap.items()} for k, fmap in state["feature_means"].items()}
        self.feature_vars = {int(k): {f: float(v) for f, v in fmap.items()} for k, fmap in state["feature_vars"].items()}
        self.is_fitted = True

    def save(self, filepath: Path) -> None:
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(self.export_state_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Path) -> "SafeNaiveBayesModel":
        with filepath.open("r", encoding="utf-8") as f:
            state = json.load(f)
        model = cls()
        model.load_state_dict(state)
        return model


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or (Path(__file__).resolve().parent / "fixture.json")
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline() -> Dict[str, Any]:
    payload = load_fixture()
    records = payload["records"]
    split = payload["metadata"]["stratified_split"]
    by_id = {r["record_id"]: r for r in records}

    train_records = [by_id[rid] for rid in split["train"]]
    test_records = [by_id[rid] for rid in split["test"]]

    # 1. Fit original model
    model = SafeNaiveBayesModel()
    model.fit(train_records)
    preds_before = model.predict(test_records)

    # 2. Save serialized state to JSON
    saved_model_path = Path(__file__).resolve().parent / "saved_pipeline.json"
    model.save(saved_model_path)

    # 3. Reload into a clean instance without training data
    reloaded_model = SafeNaiveBayesModel.load(saved_model_path)
    preds_after = reloaded_model.predict(test_records)

    # 4. Assert strict prediction parity
    agreement = preds_before == preds_after

    return {
        "task_id": "R41-T020",
        "predictions_agree": agreement,
        "test_ids": [r["record_id"] for r in test_records],
        "preds_before": preds_before,
        "preds_after": preds_after,
        "model_file": str(saved_model_path.name)
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
