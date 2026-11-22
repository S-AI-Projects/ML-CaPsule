"""Saved-model interface for the R41-T050 rainfall time-series fixture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
import json
import math
from pathlib import Path
from typing import Any

RECORD_FIELDS = {"record_id", "timestamp", "value"}
SPLIT_NAMES = ("train", "validation", "test")
SCHEMA_VERSION = 1


class TimeSeriesValidationError(ValueError):
    """Raised for fixture, configuration, schema, or model-contract violations."""


def validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise TimeSeriesValidationError("records must be a non-empty sequence")
    seen: set[str] = set()
    previous: date | None = None
    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or set(row) != RECORD_FIELDS:
            raise TimeSeriesValidationError(
                f"record {index} must contain exactly {sorted(RECORD_FIELDS)}"
            )
        record_id = row["record_id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise TimeSeriesValidationError(f"record {index} has invalid record_id")
        if record_id in seen:
            raise TimeSeriesValidationError(f"duplicate record_id: {record_id}")
        seen.add(record_id)
        timestamp = row["timestamp"]
        if not isinstance(timestamp, str):
            raise TimeSeriesValidationError(f"record {record_id} has invalid timestamp")
        try:
            current = date.fromisoformat(timestamp)
        except ValueError as error:
            raise TimeSeriesValidationError(
                f"record {record_id} timestamp must be YYYY-MM-DD"
            ) from error
        if previous is not None and current <= previous:
            raise TimeSeriesValidationError("timestamps must be consecutive and strictly increasing")
        previous = current
        value = row["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TimeSeriesValidationError(f"record {record_id} has invalid value")
        if not math.isfinite(value):
            raise TimeSeriesValidationError(f"record {record_id} has non-finite value")


def validate_splits(records: Sequence[Mapping[str, Any]], splits: Mapping[str, Sequence[str]]) -> None:
    if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_NAMES):
        raise TimeSeriesValidationError("splits must define train, validation, and test")
    all_ids = {row["record_id"] for row in records}
    assigned: list[str] = []
    positions = {row["record_id"]: index for index, row in enumerate(records)}
    split_positions: dict[str, list[int]] = {}
    for split_name in SPLIT_NAMES:
        ids = splits[split_name]
        if isinstance(ids, (str, bytes)) or not isinstance(ids, Sequence) or not ids:
            raise TimeSeriesValidationError(f"{split_name} IDs must be a non-empty sequence")
        if any(not isinstance(record_id, str) for record_id in ids):
            raise TimeSeriesValidationError(f"{split_name} IDs must be strings")
        if len(ids) != len(set(ids)):
            raise TimeSeriesValidationError(f"{split_name} contains duplicate IDs")
        if any(record_id not in all_ids for record_id in ids):
            raise TimeSeriesValidationError(f"unknown record_id in {split_name}")
        split_positions[split_name] = [positions[record_id] for record_id in ids]
        assigned.extend(ids)
    if len(assigned) != len(set(assigned)) or set(assigned) != all_ids:
        raise TimeSeriesValidationError("splits must partition every record exactly once")
    for split_name in SPLIT_NAMES:
        if split_positions[split_name] != sorted(split_positions[split_name]):
            raise TimeSeriesValidationError(f"{split_name} IDs must preserve time order")
    for earlier, later in zip(SPLIT_NAMES, SPLIT_NAMES[1:]):
        if max(split_positions[earlier]) >= min(split_positions[later]):
            raise TimeSeriesValidationError("splits must remain consecutive and unshuffled")


def fit_configuration(records: Sequence[Mapping[str, Any]], splits: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    validate_records(records)
    validate_splits(records, splits)
    by_id = {row["record_id"]: row for row in records}
    train_values = sorted(float(by_id[record_id]["value"]) for record_id in splits["train"])
    middle = len(train_values) // 2
    if len(train_values) % 2:
        forecast = train_values[middle]
    else:
        forecast = (train_values[middle - 1] + train_values[middle]) / 2.0
    return {
        "schema_version": SCHEMA_VERSION,
        "model_type": "training_median",
        "seed": 50,
        "feature_schema": ["record_id", "timestamp", "value"],
        "forecast_value": forecast,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if not isinstance(config, Mapping):
        raise TimeSeriesValidationError("configuration must be an object")
    required = {"schema_version", "model_type", "seed", "feature_schema", "forecast_value"}
    if set(config) != required:
        raise TimeSeriesValidationError("configuration has missing or unknown fields")
    if type(config["schema_version"]) is not int or config["schema_version"] != SCHEMA_VERSION:
        raise TimeSeriesValidationError("unsupported configuration schema_version")
    if config["model_type"] != "training_median":
        raise TimeSeriesValidationError("unsupported model_type")
    if type(config["seed"]) is not int or config["seed"] != 50:
        raise TimeSeriesValidationError("unsupported seed")
    if config["feature_schema"] != ["record_id", "timestamp", "value"]:
        raise TimeSeriesValidationError("feature_schema mismatch")
    forecast = config["forecast_value"]
    if isinstance(forecast, bool) or not isinstance(forecast, (int, float)):
        raise TimeSeriesValidationError("forecast_value must be numeric")
    if not math.isfinite(forecast):
        raise TimeSeriesValidationError("forecast_value must be finite")


def save_config(config: Mapping[str, Any], path: str | Path) -> None:
    validate_config(config)
    Path(path).write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as source:
        config = json.load(source)
    validate_config(config)
    return config


def validate_prediction_input(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping) or set(record) != RECORD_FIELDS:
        raise TimeSeriesValidationError("prediction record must contain exactly record_id, timestamp, value")
    record_id = record["record_id"]
    if not isinstance(record_id, str) or not record_id.strip():
        raise TimeSeriesValidationError("prediction record_id is invalid")
    timestamp = record["timestamp"]
    if not isinstance(timestamp, str):
        raise TimeSeriesValidationError("prediction timestamp is invalid")
    try:
        date.fromisoformat(timestamp)
    except ValueError as error:
        raise TimeSeriesValidationError("prediction timestamp must be YYYY-MM-DD") from error
    value = record["value"]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TimeSeriesValidationError("prediction value must be numeric")
    if not math.isfinite(value):
        raise TimeSeriesValidationError("prediction value must be finite")


def predict(record: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    validate_prediction_input(record)
    validate_config(config)
    return float(config["forecast_value"])


def load_fixture() -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    fixture_path = Path(__file__).with_name("fixture.json")
    with fixture_path.open(encoding="utf-8") as source:
        fixture = json.load(source)
    if fixture.get("data_kind") != "simulated" or fixture.get("fixture_kind") != "deterministic_timeseries":
        raise TimeSeriesValidationError("fixture must be explicitly marked simulated deterministic_timeseries")
    if fixture.get("seed") != 50:
        raise TimeSeriesValidationError("fixture seed must be 50")
    validate_records(fixture["records"])
    validate_splits(fixture["records"], fixture["split_ids"])
    return fixture["records"], fixture["split_ids"]


if __name__ == "__main__":
    fixture_records, fixture_splits = load_fixture()
    fixture_config = fit_configuration(fixture_records, fixture_splits)
    print(json.dumps(fixture_config, indent=2, sort_keys=True))
