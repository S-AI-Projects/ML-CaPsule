"""Explicit numeric preprocessing & validation pipeline for Glass Identification (Task R41-T015)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ExplicitNumericPreprocessor:
    """Standardizes numeric features using statistics fitted exclusively on training rows."""

    def __init__(self, feature_order: List[str], valid_labels: List[int]) -> None:
        self.feature_order = list(feature_order)
        self.valid_labels = set(valid_labels)
        self.is_fitted = False
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.mins: Dict[str, float] = {}
        self.maxs: Dict[str, float] = {}

    def _validate_record(self, record: Dict[str, Any], check_label: bool = True) -> None:
        for f in self.feature_order:
            if f not in record:
                raise ValueError(f"Missing required feature: {f}")
            val = record[f]
            if val is None or not isinstance(val, (int, float)):
                raise ValueError(f"Feature {f} must be numeric, got {val}")

        if check_label and "label" in record:
            lbl = record["label"]
            if lbl not in self.valid_labels:
                raise ValueError(f"Invalid label {lbl}, expected one of {self.valid_labels}")

    def fit(self, training_records: List[Dict[str, Any]]) -> "ExplicitNumericPreprocessor":
        if not training_records:
            raise ValueError("Cannot fit preprocessor on empty dataset")

        for r in training_records:
            self._validate_record(r, check_label=True)

        n = len(training_records)
        self.means = {}
        self.stds = {}
        self.mins = {}
        self.maxs = {}

        for f in self.feature_order:
            vals = [float(r[f]) for r in training_records]
            mean_val = sum(vals) / n
            # Sample standard deviation (or 1.0 fallback if constant)
            var = sum((v - mean_val) ** 2 for v in vals) / (n - 1) if n > 1 else 0.0
            std_val = var ** 0.5 if var > 0 else 1.0

            self.means[f] = round(mean_val, 4)
            self.stds[f] = round(std_val, 4)
            self.mins[f] = round(min(vals), 4)
            self.maxs[f] = round(max(vals), 4)

        self.is_fitted = True
        return self

    def transform(self, records: List[Dict[str, Any]], check_label: bool = False) -> List[List[float]]:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before transforming")
        if not records:
            return []

        # Transform interface applies frozen training statistics; cannot re-fit or adapt to held-out data
        transformed = []
        for r in records:
            self._validate_record(r, check_label=check_label)
            row = []
            for f in self.feature_order:
                # Standard scaling: (x - mean) / std
                scaled = (float(r[f]) - self.means[f]) / self.stds[f]
                row.append(round(scaled, 4))
            transformed.append(row)
        return transformed

    def fit_transform(self, training_records: List[Dict[str, Any]]) -> List[List[float]]:
        return self.fit(training_records).transform(training_records)


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or (Path(__file__).resolve().parent / "fixture.json")
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline() -> Dict[str, Any]:
    payload = load_fixture()
    metadata = payload["metadata"]
    records = payload["records"]
    by_id = {r["record_id"]: r for r in records}

    train_records = [by_id[i] for i in metadata["stratified_split"]["train"]]
    val_records = [by_id[i] for i in metadata["stratified_split"]["validation"]]
    test_records = [by_id[i] for i in metadata["stratified_split"]["test"]]

    preprocessor = ExplicitNumericPreprocessor(
        feature_order=metadata["feature_order"],
        valid_labels=metadata["valid_labels"],
    )

    # Strictly fit on training set
    preprocessor.fit(train_records)
    train_transformed = preprocessor.transform(train_records)
    val_transformed = preprocessor.transform(val_records)
    test_transformed = preprocessor.transform(test_records)

    return {
        "task_id": "R41-T015",
        "fitted_means": preprocessor.means,
        "fitted_stds": preprocessor.stds,
        "train_shape": [len(train_transformed), len(metadata["feature_order"])],
        "validation_shape": [len(val_transformed), len(metadata["feature_order"])],
        "test_shape": [len(test_transformed), len(metadata["feature_order"])],
        "train_scaled_sample": train_transformed,
        "test_scaled_sample": test_transformed,
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
