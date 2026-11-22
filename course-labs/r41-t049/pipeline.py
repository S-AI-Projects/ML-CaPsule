"""Offline robustness fixture for Avocado-prices-style numeric regression."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

RECORD_FIELDS = {"record_id", "x1", "x2", "target"}
FEATURES = ("x1", "x2")
SPLIT_NAMES = ("train", "validation", "test")


class RegressionValidationError(ValueError):
    """Raised for data, split, or shape contract violations."""


def validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise RegressionValidationError("records must be a non-empty sequence")
    seen: set[str] = set()
    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or set(row) != RECORD_FIELDS:
            raise RegressionValidationError(
                f"record {index} must contain exactly {sorted(RECORD_FIELDS)}"
            )
        record_id = row["record_id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise RegressionValidationError(f"record {index} has invalid record_id")
        if record_id in seen:
            raise RegressionValidationError(f"duplicate record_id: {record_id}")
        seen.add(record_id)
        for field in ("x1", "x2", "target"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RegressionValidationError(f"record {record_id} has invalid {field}")
            if not math.isfinite(value):
                raise RegressionValidationError(f"record {record_id} has non-finite {field}")


def validate_splits(records: Sequence[Mapping[str, Any]], splits: Mapping[str, Sequence[str]]) -> None:
    if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_NAMES):
        raise RegressionValidationError("splits must define train, validation, and test")
    all_ids = {row["record_id"] for row in records}
    assigned: list[str] = []
    for split_name in SPLIT_NAMES:
        ids = splits[split_name]
        if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence) or not ids:
            raise RegressionValidationError(f"{split_name} IDs must be a non-empty sequence")
        if any(not isinstance(record_id, str) for record_id in ids):
            raise RegressionValidationError(f"{split_name} IDs must be strings")
        if len(ids) != len(set(ids)):
            raise RegressionValidationError(f"{split_name} contains duplicate IDs")
        if any(record_id not in all_ids for record_id in ids):
            raise RegressionValidationError(f"unknown record_id in {split_name}")
        assigned.extend(ids)
    if len(assigned) != len(set(assigned)) or set(assigned) != all_ids:
        raise RegressionValidationError("splits must partition every record exactly once")


def fit_linear(training: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """Fit intercept and x1 coefficient from training rows only."""
    x_values = [float(row["x1"]) for row in training]
    y_values = [float(row["target"]) for row in training]
    count = len(training)
    x_mean = sum(x_values) / count
    y_mean = sum(y_values) / count
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if abs(denominator) < 1e-12:
        raise RegressionValidationError("training x1 feature has no variance")
    weight = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values)
    ) / denominator
    intercept = y_mean - weight * x_mean
    return intercept, weight


def feature_vector(record: Mapping[str, Any]) -> list[float]:
    if not isinstance(record, Mapping) or set(record) != RECORD_FIELDS:
        raise RegressionValidationError("record must contain exactly the documented fields")
    values: list[float] = []
    for field in FEATURES:
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RegressionValidationError(f"feature {field} must be numeric")
        if not math.isfinite(value):
            raise RegressionValidationError(f"feature {field} must be finite")
        values.append(float(value))
    return values


def predict(record: Mapping[str, Any], coefficients: tuple[float, float]) -> float:
    x1, _ = feature_vector(record)
    intercept, weight_x1 = coefficients
    return intercept + weight_x1 * x1


def run_pipeline(records: Sequence[Mapping[str, Any]], splits: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    validate_records(records)
    validate_splits(records, splits)
    by_id = {row["record_id"]: row for row in records}
    training = [by_id[record_id] for record_id in splits["train"]]
    coefficients = fit_linear(training)

    predictions: dict[str, dict[str, float]] = {}
    for split_name in ("validation", "test"):
        for record_id in splits[split_name]:
            row = by_id[record_id]
            predicted = predict(row, coefficients)
            predictions[record_id] = {
                "target": float(row["target"]),
                "predicted": predicted,
                "absolute_error": abs(predicted - float(row["target"])),
            }
    return {
        "data_kind": "simulated_pipeline_verification",
        "fixture_kind": "deterministic_formula",
        "seed": 49,
        "coefficients": [coefficients[0], coefficients[1]],
        "train_record_ids": list(splits["train"]),
        "predictions": predictions,
    }


def load_fixture() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    fixture_path = Path(__file__).with_name("fixture.json")
    with fixture_path.open(encoding="utf-8") as source:
        fixture = json.load(source)
    if fixture.get("data_kind") != "simulated" or fixture.get("fixture_kind") != "deterministic_formula":
        raise RegressionValidationError("fixture must be explicitly marked simulated deterministic_formula")
    if fixture.get("seed") != 49:
        raise RegressionValidationError("fixture seed must be 49")
    validate_records(fixture["records"])
    validate_splits(fixture["records"], fixture["split_ids"])
    return fixture["records"], fixture["split_ids"]


if __name__ == "__main__":
    fixture_records, fixture_splits = load_fixture()
    print(json.dumps(run_pipeline(fixture_records, fixture_splits), indent=2, sort_keys=True))
