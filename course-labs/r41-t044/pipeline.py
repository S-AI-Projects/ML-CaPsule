"""Fixture validation and metric checks for R41-T044."""

import json
import math
from pathlib import Path
from typing import Any


FIXTURE_PATH = Path(__file__).with_name("fixture.json")
RECORD_FIELDS = {"record_id", "group_id", "pixels", "label"}
SPLIT_NAMES = ("train", "validation", "test")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_records(records: Any) -> None:
    """Validate the declared image-row contract without coercing input values."""
    _require(isinstance(records, list) and bool(records), "records must be a non-empty list")
    seen_ids = set()
    for index, record in enumerate(records):
        _require(
            isinstance(record, dict) and set(record) == RECORD_FIELDS,
            f"record {index} must contain exactly {sorted(RECORD_FIELDS)}",
        )
        for field in ("record_id", "group_id"):
            value = record[field]
            _require(
                isinstance(value, str) and bool(value.strip()),
                f"record {index} has an invalid {field}",
            )
        _require(record["record_id"] not in seen_ids, f"duplicate record_id: {record['record_id']}")
        seen_ids.add(record["record_id"])

        pixels = record["pixels"]
        _require(
            isinstance(pixels, list)
            and len(pixels) == 4
            and all(isinstance(row, list) and len(row) == 4 for row in pixels),
            f"record {record['record_id']} pixels must be a 4x4 list",
        )
        for row in pixels:
            for value in row:
                _require(
                    type(value) is int and 0 <= value <= 255,
                    f"record {record['record_id']} pixels must be integer values from 0 to 255",
                )
        _require(
            type(record["label"]) is int and record["label"] in (0, 1),
            f"record {record['record_id']} label must be integer 0 or 1",
        )


def validate_splits(records: list[dict[str, Any]], split_group_ids: Any) -> dict[str, list[str]]:
    """Return record IDs by split after enforcing complete group-level isolation."""
    _require(
        isinstance(split_group_ids, dict) and set(split_group_ids) == set(SPLIT_NAMES),
        "split_group_ids must declare train, validation, and test",
    )
    group_to_records: dict[str, list[str]] = {}
    for record in records:
        group_to_records.setdefault(record["group_id"], []).append(record["record_id"])

    assigned_groups = []
    split_record_ids = {}
    for split_name in SPLIT_NAMES:
        groups = split_group_ids[split_name]
        _require(
            isinstance(groups, list)
            and bool(groups)
            and all(isinstance(group, str) and group for group in groups),
            f"{split_name} must contain declared group IDs",
        )
        _require(
            len(groups) == len(set(groups)),
            f"{split_name} contains duplicate group IDs",
        )
        assigned_groups.extend(groups)
        unknown = set(groups) - set(group_to_records)
        _require(not unknown, f"{split_name} has unknown group IDs: {sorted(unknown)}")
        split_record_ids[split_name] = [
            record_id
            for group in groups
            for record_id in group_to_records[group]
        ]

    _require(
        len(assigned_groups) == len(set(assigned_groups)),
        "groups must not overlap between train, validation, and test",
    )
    _require(
        set(assigned_groups) == set(group_to_records),
        "every fixture group must belong to exactly one split",
    )
    return split_record_ids


def load_fixture(path: str | Path = FIXTURE_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)
    _require(isinstance(fixture, dict), "fixture must be a JSON object")
    _require(fixture.get("data_kind") == "simulated", "fixture must be marked simulated")
    validate_records(fixture.get("records"))
    validate_splits(fixture["records"], fixture.get("split_group_ids"))
    return fixture


def _binary_labels(values: Any, argument_name: str) -> list[int]:
    _require(
        isinstance(values, (list, tuple)) and bool(values),
        f"{argument_name} must be a non-empty list or tuple",
    )
    _require(
        all(type(value) is int and value in (0, 1) for value in values),
        f"{argument_name} must contain only integer labels 0 or 1",
    )
    return list(values)


def binary_classification_metrics(y_true: Any, y_pred: Any) -> dict[str, Any]:
    """Compute fixed-label macro precision/recall; an undefined class score is 0."""
    truth = _binary_labels(y_true, "y_true")
    predictions = _binary_labels(y_pred, "y_pred")
    _require(len(truth) == len(predictions), "y_true and y_pred lengths must match")

    per_class = {}
    for label in (0, 1):
        true_positive = sum(
            actual == label and predicted == label
            for actual, predicted in zip(truth, predictions)
        )
        false_positive = sum(
            actual != label and predicted == label
            for actual, predicted in zip(truth, predictions)
        )
        false_negative = sum(
            actual == label and predicted != label
            for actual, predicted in zip(truth, predictions)
        )
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        per_class[str(label)] = {
            "precision": (
                true_positive / precision_denominator if precision_denominator else 0.0
            ),
            "recall": true_positive / recall_denominator if recall_denominator else 0.0,
        }

    return {
        "averaging": "macro over fixed labels [0, 1]",
        "zero_denominator": 0.0,
        "per_class": per_class,
        "precision": math.fsum(
            per_class[str(label)]["precision"] for label in (0, 1)
        )
        / 2,
        "recall": math.fsum(per_class[str(label)]["recall"] for label in (0, 1)) / 2,
    }


def mean_absolute_error(y_true: Any, y_pred: Any) -> float:
    """Return mean absolute error; inputs must be non-empty, equal-length vectors."""
    _require(
        isinstance(y_true, (list, tuple)) and bool(y_true),
        "y_true must be a non-empty list or tuple",
    )
    _require(
        isinstance(y_pred, (list, tuple)) and bool(y_pred),
        "y_pred must be a non-empty list or tuple",
    )
    _require(len(y_true) == len(y_pred), "y_true and y_pred lengths must match")
    values = list(y_true) + list(y_pred)
    _require(
        all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in values
        ),
        "regression values must be finite real numbers",
    )
    return math.fsum(abs(actual - predicted) for actual, predicted in zip(y_true, y_pred)) / len(y_true)


def run_examples() -> dict[str, Any]:
    fixture = load_fixture()
    return {
        "fixture": {
            "data_kind": fixture["data_kind"],
            "record_count": len(fixture["records"]),
            "split_record_ids": validate_splits(
                fixture["records"], fixture["split_group_ids"]
            ),
        },
        "classification_example": {
            "y_true": [1, 1, 0, 0],
            "y_pred": [1, 0, 1, 0],
            "metrics": binary_classification_metrics([1, 1, 0, 0], [1, 0, 1, 0]),
        },
        "regression_example": {
            "y_true": [1, 2, 3],
            "y_pred": [1, 3, 2],
            "mae": mean_absolute_error([1, 2, 3], [1, 3, 2]),
        },
        "result_scope": "synthetic pipeline verification only; no model performance claim",
    }


if __name__ == "__main__":
    print(json.dumps(run_examples(), indent=2))
