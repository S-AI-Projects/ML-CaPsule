"""Error analysis for the R41-T048 deterministic formula fixture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any


RECORD_FIELDS = {"record_id", "x1", "x2", "target"}
SPLIT_NAMES = ("train", "validation", "test")


class FixtureValidationError(ValueError):
    """Raised when fixture inputs violate the documented data contract."""


def validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise FixtureValidationError("records must be a non-empty sequence")

    seen: set[str] = set()
    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or set(row) != RECORD_FIELDS:
            raise FixtureValidationError(
                f"record {index} must contain exactly {sorted(RECORD_FIELDS)}"
            )
        record_id = row["record_id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise FixtureValidationError(f"record {index} has an invalid record_id")
        if record_id in seen:
            raise FixtureValidationError(f"duplicate record_id: {record_id}")
        seen.add(record_id)

        for field in ("x1", "x2", "target"):
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise FixtureValidationError(f"record {record_id} has invalid {field}")
            if not math.isfinite(value):
                raise FixtureValidationError(f"record {record_id} has non-finite {field}")


def validate_splits(
    records: Sequence[Mapping[str, Any]], splits: Mapping[str, Sequence[str]]
) -> None:
    if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_NAMES):
        raise FixtureValidationError("splits must define train, validation, and test")
    all_ids = {row["record_id"] for row in records}
    assigned: list[str] = []
    for split_name in SPLIT_NAMES:
        ids = splits[split_name]
        if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence) or not ids:
            raise FixtureValidationError(f"{split_name} IDs must be a non-empty sequence")
        if any(not isinstance(record_id, str) for record_id in ids):
            raise FixtureValidationError(f"{split_name} IDs must be strings")
        if len(ids) != len(set(ids)):
            raise FixtureValidationError(f"{split_name} contains duplicate IDs")
        assigned.extend(ids)
    if len(assigned) != len(set(assigned)) or set(assigned) != all_ids:
        raise FixtureValidationError("splits must partition every record exactly once")


def analyze_errors(
    records: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, float],
    splits: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Create per-row error entries and split aggregates from supplied predictions."""
    validate_records(records)
    validate_splits(records, splits)
    if not isinstance(predictions, Mapping):
        raise FixtureValidationError("predictions must map record IDs to numeric values")
    record_ids = {row["record_id"] for row in records}
    if set(predictions) != record_ids:
        raise FixtureValidationError("predictions must contain exactly the fixture record IDs")

    normalized_predictions: dict[str, float] = {}
    for record_id, predicted in predictions.items():
        if isinstance(predicted, bool) or not isinstance(predicted, (int, float)):
            raise FixtureValidationError(f"prediction for {record_id} must be numeric")
        if not math.isfinite(predicted):
            raise FixtureValidationError(f"prediction for {record_id} must be finite")
        normalized_predictions[record_id] = float(predicted)

    split_by_id = {
        record_id: split_name
        for split_name, ids in splits.items()
        for record_id in ids
    }
    entries = []
    for row in records:
        record_id = row["record_id"]
        expected = float(row["target"])
        predicted = normalized_predictions[record_id]
        residual = predicted - expected
        entries.append(
            {
                "record_id": record_id,
                "split": split_by_id[record_id],
                "expected_value": expected,
                "predicted_value": predicted,
                "absolute_error": abs(residual),
                "reason_category": (
                    "no_error"
                    if residual == 0
                    else "overprediction"
                    if residual > 0
                    else "underprediction"
                ),
            }
        )

    aggregates = {}
    for split_name in SPLIT_NAMES:
        split_entries = [entry for entry in entries if entry["split"] == split_name]
        errors = [entry for entry in split_entries if entry["reason_category"] != "no_error"]
        aggregates[split_name] = {
            "record_count": len(split_entries),
            "error_count": len(errors),
            "no_error_count": len(split_entries) - len(errors),
        }

    return {
        "data_kind": "simulated_pipeline_verification",
        "fixture_kind": "deterministic_formula",
        "records": entries,
        "aggregates": aggregates,
        "total_record_count": len(entries),
        "total_error_count": sum(
            aggregate["error_count"] for aggregate in aggregates.values()
        ),
    }


def load_fixture() -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, float]]:
    path = Path(__file__).with_name("fixture.json")
    with path.open(encoding="utf-8") as source:
        fixture = json.load(source)
    if fixture.get("data_kind") != "simulated":
        raise FixtureValidationError("fixture must be explicitly marked simulated")
    records = fixture["records"]
    validate_records(records)
    validate_splits(records, fixture["split_ids"])
    predictions = {
        row["record_id"]: float(2 * row["x1"] + 1) for row in records
    }
    predictions["r6"] += 1.0
    return records, fixture["split_ids"], predictions


if __name__ == "__main__":
    records, splits, predictions = load_fixture()
    print(
        json.dumps(
            analyze_errors(records, predictions, splits),
            indent=2,
            sort_keys=True,
        )
    )
