"""Small deterministic clustering error-analysis fixture for R41-T058."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Mapping, Sequence


FIXTURE_SCHEMA = "r41-t058.fixture.v1"
RECORD_FIELDS = {"record_id", "x1", "x2"}
CENTROIDS = {0: (0.0, 0.0), 1: (5.0, 5.0)}


def validate_records(records: object) -> list[dict]:
    """Validate stable IDs and two-dimensional finite numeric points."""
    if not isinstance(records, list):
        raise ValueError("records must be a list")

    validated = []
    seen_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("record at index %d must be an object" % index)
        if set(record) != RECORD_FIELDS:
            raise ValueError(
                "record at index %d must contain exactly record_id, x1, x2" % index
            )

        record_id = record["record_id"]
        x1 = record["x1"]
        x2 = record["x2"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        if record_id in seen_ids:
            raise ValueError("duplicate record_id: %s" % record_id)
        for field, value in (("x1", x1), ("x2", x2)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("%s must be a finite number" % field)
            if not math.isfinite(value):
                raise ValueError("%s must be a finite number" % field)

        seen_ids.add(record_id)
        validated.append({"record_id": record_id, "x1": x1, "x2": x2})
    return validated


def load_fixture(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("fixture schema mismatch: expected %s" % FIXTURE_SCHEMA)
    if payload.get("simulated") is not True:
        raise ValueError("fixture must be explicitly marked simulated")
    records = validate_records(payload.get("records"))
    if len(records) != 6:
        raise ValueError("the R41-T058 fixture must contain exactly six records")
    return records


def fixed_splits(
    records: Sequence[Mapping[str, object]], seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    """Create reproducible 4/1/1 train, validation, and test partitions."""
    validated = validate_records([dict(record) for record in records])
    if len(validated) != 6:
        raise ValueError("fixed splits require exactly six records")
    shuffled = list(validated)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:4], shuffled[4:5], shuffled[5:]


def assign_nearest_centroid(
    records: Sequence[Mapping[str, object]],
    centroids: Mapping[int, Sequence[float]] = CENTROIDS,
) -> dict[str, int]:
    """Assign each point to its closest fixed centroid; ties choose the lower ID."""
    validated = validate_records([dict(record) for record in records])
    if not isinstance(centroids, Mapping) or not centroids:
        raise ValueError("centroids must be a non-empty mapping")

    checked_centroids = {}
    for label, coordinates in centroids.items():
        if type(label) is not int:
            raise ValueError("centroid IDs must be integers")
        if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 2:
            raise ValueError("each centroid must contain exactly two coordinates")
        checked = []
        for coordinate in coordinates:
            if isinstance(coordinate, bool) or not isinstance(coordinate, (int, float)):
                raise ValueError("centroid coordinates must be finite numbers")
            if not math.isfinite(coordinate):
                raise ValueError("centroid coordinates must be finite numbers")
            checked.append(coordinate)
        checked_centroids[label] = tuple(checked)

    assignments = {}
    for record in validated:
        assignments[record["record_id"]] = min(
            (
                (record["x1"] - x1) ** 2 + (record["x2"] - x2) ** 2,
                label,
            )
            for label, (x1, x2) in checked_centroids.items()
        )[1]
    return assignments


def build_error_report(
    records: Sequence[Mapping[str, object]],
    predictions: Mapping[str, int],
    centroids: Mapping[int, Sequence[float]] = CENTROIDS,
) -> dict:
    """Report only incorrect assignments with ID, expected/predicted values, reason."""
    validated = validate_records([dict(record) for record in records])
    expected = assign_nearest_centroid(validated, centroids)
    if not isinstance(predictions, Mapping):
        raise ValueError("predictions must map record IDs to integer cluster IDs")
    if set(predictions) != set(expected):
        raise ValueError("prediction IDs must exactly match fixture record IDs")

    errors = []
    for record in validated:
        record_id = record["record_id"]
        predicted = predictions[record_id]
        if type(predicted) is not int or predicted not in centroids:
            raise ValueError("predicted cluster IDs must match configured centroid IDs")
        if predicted != expected[record_id]:
            errors.append(
                {
                    "record_id": record_id,
                    "expected_value": expected[record_id],
                    "predicted_value": predicted,
                    "reason_category": "cluster_assignment_mismatch",
                }
            )

    errors_by_reason = {}
    for row in errors:
        reason = row["reason_category"]
        errors_by_reason[reason] = errors_by_reason.get(reason, 0) + 1
    return {
        "rows": errors,
        "summary": {
            "record_count": len(validated),
            "correct_count": len(validated) - len(errors),
            "error_count": len(errors),
            "errors_by_reason": errors_by_reason,
        },
    }


def load_predictions(path: str | Path) -> dict[str, int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema") != "r41-t058.predictions.v1":
        raise ValueError("prediction schema mismatch")
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")
    result = {}
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict) or set(prediction) != {
            "record_id",
            "predicted_value",
        }:
            raise ValueError(
                "prediction at index %d must contain exactly record_id, predicted_value"
                % index
            )
        record_id = prediction["record_id"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("prediction record_id must be a non-empty string")
        if record_id in result:
            raise ValueError("duplicate prediction record_id: %s" % record_id)
        result[record_id] = prediction["predicted_value"]
    return result
