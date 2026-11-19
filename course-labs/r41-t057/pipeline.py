"""Serializable deterministic time-series configuration fixture for R41-T057."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping, Sequence


FIXTURE_SCHEMA = "r41-t057.fixture.v1"
CONFIG_SCHEMA = "r41-t057.config.v1"
CONFIG_FIELDS = {
    "schema_version",
    "seed",
    "feature_order",
    "split_ids",
    "algorithm",
}
SPLIT_NAMES = ("train", "validation", "test")


def validate_records(records: object) -> list[dict]:
    """Validate the record schema and return records in timestamp order."""
    if not isinstance(records, list):
        raise ValueError("records must be a list")

    validated = []
    seen_ids = set()
    seen_timestamps = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("record at index %d must be an object" % index)
        if set(record) != {"record_id", "timestamp", "value"}:
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

    ordered = sorted(validated, key=lambda record: record["timestamp"])
    if len(ordered) != 10:
        raise ValueError("the fixture must contain exactly ten daily records")
    for previous, current in zip(ordered, ordered[1:]):
        previous_date = date.fromisoformat(previous["timestamp"])
        current_date = date.fromisoformat(current["timestamp"])
        if current_date - previous_date != timedelta(days=1):
            raise ValueError("records must be ten consecutive daily observations")
    return ordered


def load_fixture(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("fixture schema mismatch: expected %s" % FIXTURE_SCHEMA)
    if payload.get("simulated") is not True:
        raise ValueError("fixture must be explicitly marked simulated")
    return validate_records(payload.get("records"))


def expected_splits(records: Sequence[Mapping[str, object]]) -> dict[str, list[str]]:
    ordered = validate_records([dict(record) for record in records])
    return {
        "train": [record["record_id"] for record in ordered[:6]],
        "validation": [record["record_id"] for record in ordered[6:8]],
        "test": [record["record_id"] for record in ordered[8:]],
    }


def create_config(
    records: Sequence[Mapping[str, object]], seed: int = 42
) -> dict:
    """Create the only supported configuration, preserving chronological splits."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return {
        "schema_version": CONFIG_SCHEMA,
        "seed": seed,
        "feature_order": ["value"],
        "split_ids": expected_splits(records),
        "algorithm": {"name": "last_value", "window_size": 1},
    }


def validate_config(
    config: object, records: Sequence[Mapping[str, object]]
) -> dict:
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    if set(config) != CONFIG_FIELDS:
        raise ValueError("configuration contains unknown or missing fields")
    if config["schema_version"] != CONFIG_SCHEMA:
        raise ValueError("unsupported configuration schema version")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if config["feature_order"] != ["value"]:
        raise ValueError("feature_order must be exactly ['value']")

    split_ids = config["split_ids"]
    expected = expected_splits(records)
    if not isinstance(split_ids, dict) or set(split_ids) != set(SPLIT_NAMES):
        raise ValueError("split_ids must define train, validation, and test")
    if any(not isinstance(split_ids[name], list) for name in SPLIT_NAMES):
        raise ValueError("each split must be a list of record IDs")
    if split_ids != expected:
        raise ValueError("split IDs must match the chronological 6/2/2 partition")

    algorithm = config["algorithm"]
    if not isinstance(algorithm, dict) or set(algorithm) != {"name", "window_size"}:
        raise ValueError("algorithm contains unknown or missing settings")
    if algorithm["name"] != "last_value" or type(algorithm["window_size"]) is not int:
        raise ValueError("unsupported algorithm settings")
    if algorithm["window_size"] != 1:
        raise ValueError("last_value requires window_size 1")
    return {
        "schema_version": config["schema_version"],
        "seed": config["seed"],
        "feature_order": list(config["feature_order"]),
        "split_ids": {
            name: list(split_ids[name]) for name in SPLIT_NAMES
        },
        "algorithm": dict(algorithm),
    }


def save_config(
    config: Mapping[str, object],
    path: str | Path,
    records: Sequence[Mapping[str, object]],
) -> None:
    validated = validate_config(dict(config), records)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(validated, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_config(path: str | Path, records: Sequence[Mapping[str, object]]) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return validate_config(config, records)


def run_pipeline(
    records: Sequence[Mapping[str, object]], config: Mapping[str, object]
) -> dict:
    """Return canonical split IDs and last-training-value forecasts."""
    ordered = validate_records([dict(record) for record in records])
    validated_config = validate_config(dict(config), ordered)
    records_by_id = {record["record_id"]: record for record in ordered}
    split_ids = validated_config["split_ids"]
    training = [records_by_id[record_id] for record_id in split_ids["train"]]
    targets = [
        records_by_id[record_id]
        for split_name in ("validation", "test")
        for record_id in split_ids[split_name]
    ]
    if any(
        target["timestamp"] <= training[-1]["timestamp"] for target in targets
    ):
        raise ValueError("validation and test records must follow the training period")
    predictions = {
        record["record_id"]: training[-1]["value"] for record in targets
    }
    return {"split_ids": split_ids, "predictions": predictions}
