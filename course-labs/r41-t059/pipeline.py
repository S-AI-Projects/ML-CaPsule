"""Chronological, standard-library-only forecasting fixture for R41-T059."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path


FIXTURE_SCHEMA = "r41-t059.fixture.v1"
RECORD_FIELDS = {"record_id", "timestamp", "value"}


def validate_records(records: object) -> list[dict]:
    """Validate record shape and types without changing time order."""
    if not isinstance(records, list):
        raise ValueError("records must be a list")

    validated = []
    seen_ids = set()
    seen_timestamps = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("record at index %d must be an object" % index)
        if set(record) != RECORD_FIELDS:
            raise ValueError(
                "record at index %d must contain exactly record_id, timestamp, value"
                % index
            )

        record_id = record["record_id"]
        timestamp = record["timestamp"]
        value = record["value"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        if not isinstance(timestamp, str):
            raise ValueError("timestamp must be an ISO date string")
        try:
            parsed_date = date.fromisoformat(timestamp)
        except ValueError as error:
            raise ValueError("timestamp must be an ISO date string") from error
        if parsed_date.isoformat() != timestamp:
            raise ValueError("timestamp must use YYYY-MM-DD format")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("value must be a finite number")
        if not math.isfinite(value):
            raise ValueError("value must be a finite number")
        if record_id in seen_ids:
            raise ValueError("duplicate record_id: %s" % record_id)
        if timestamp in seen_timestamps:
            raise ValueError("duplicate timestamp: %s" % timestamp)

        seen_ids.add(record_id)
        seen_timestamps.add(timestamp)
        validated.append(
            {"record_id": record_id, "timestamp": timestamp, "value": value}
        )
    return validated


def load_fixture(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("fixture schema mismatch: expected %s" % FIXTURE_SCHEMA)
    if payload.get("simulated") is not True:
        raise ValueError("fixture must be explicitly marked simulated")

    records = validate_records(payload.get("records"))
    fixed_splits(records)
    return records


def _ordered_daily_records(records: object) -> list[dict]:
    ordered = sorted(validate_records(records), key=lambda record: record["timestamp"])
    if len(ordered) != 10:
        raise ValueError("the fixed split requires exactly ten daily records")
    for previous, current in zip(ordered, ordered[1:]):
        previous_date = date.fromisoformat(previous["timestamp"])
        current_date = date.fromisoformat(current["timestamp"])
        if current_date - previous_date != timedelta(days=1):
            raise ValueError("records must be ten consecutive daily observations")
    return ordered


def fixed_splits(records: object) -> tuple[list[dict], list[dict], list[dict]]:
    """Return chronological 6/2/2 train, validation, and test partitions."""
    ordered = _ordered_daily_records(records)
    return ordered[:6], ordered[6:8], ordered[8:]


def forecast_last_value(
    train_records: object, target_records: object
) -> dict[str, int | float]:
    """Forecast each later target with the latest training observation."""
    training = validate_records(train_records)
    targets = validate_records(target_records)
    if not training:
        raise ValueError("train_records must not be empty")
    if not targets:
        raise ValueError("target_records must not be empty")

    training_ids = {record["record_id"] for record in training}
    if training_ids.intersection(record["record_id"] for record in targets):
        raise ValueError("training and target record IDs must be distinct")

    latest_training = max(training, key=lambda record: record["timestamp"])
    if any(record["timestamp"] <= latest_training["timestamp"] for record in targets):
        raise ValueError("all target timestamps must follow the training period")

    ordered_targets = sorted(targets, key=lambda record: record["timestamp"])
    return {
        record["record_id"]: latest_training["value"] for record in ordered_targets
    }
