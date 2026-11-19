"""Training-mean regression baseline for the R41-T053 synthetic fixture."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence


FIXTURE_SCHEMA = "r41-t053.fixture.v1"
BASELINE_SCHEMA = "r41-t053.mean-baseline.v1"
FIXED_SEED = 42
FIXED_SPLIT_IDS = {
    "train": ["r1", "r2", "r3", "r4"],
    "validation": ["r5"],
    "test": ["r6"],
}


def validate_records(records: object) -> list[dict]:
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    if len(records) > 32:
        raise ValueError("records must contain at most 32 samples")

    validated = []
    seen_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("record at index %d must be an object" % index)
        if set(record) != {"record_id", "x1", "x2", "target"}:
            raise ValueError(
                "record at index %d must contain exactly record_id, x1, x2, target"
                % index
            )
        record_id = record["record_id"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        if record_id in seen_ids:
            raise ValueError("duplicate record_id: %s" % record_id)
        checked = {"record_id": record_id}
        for field in ("x1", "x2", "target"):
            value = record[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("%s must be a finite number" % field)
            if not math.isfinite(value):
                raise ValueError("%s must be a finite number" % field)
            checked[field] = value
        seen_ids.add(record_id)
        validated.append(checked)
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
        raise ValueError("the R41-T053 fixture must contain exactly six rows")
    if [record["record_id"] for record in records] != [
        "r1", "r2", "r3", "r4", "r5", "r6"
    ]:
        raise ValueError("fixture record IDs must be ordered r1 through r6")
    if any(record["x2"] != 1 for record in records):
        raise ValueError("fixture x2 values must all equal 1")
    if any(record["target"] != 2 * record["x1"] + 1 for record in records):
        raise ValueError("fixture targets must follow target = 2*x1 + 1")
    return records


def fixed_split_ids(seed: int = FIXED_SEED) -> dict[str, list[str]]:
    if type(seed) is not int or seed != FIXED_SEED:
        raise ValueError("the fixed split uses seed 42")
    return {name: list(ids) for name, ids in FIXED_SPLIT_IDS.items()}


def _validate_split_ids(split_ids: object, seed: int) -> dict[str, list[str]]:
    if not isinstance(split_ids, dict) or set(split_ids) != set(FIXED_SPLIT_IDS):
        raise ValueError("split_ids must define train, validation, and test")
    if any(not isinstance(split_ids[name], list) for name in FIXED_SPLIT_IDS):
        raise ValueError("each split must be a list of record IDs")
    flattened = [record_id for ids in split_ids.values() for record_id in ids]
    if any(not isinstance(record_id, str) or not record_id for record_id in flattened):
        raise ValueError("split IDs must be non-empty strings")
    if len(flattened) != len(set(flattened)):
        raise ValueError("split IDs overlap; held-out contamination is not allowed")
    if split_ids != fixed_split_ids(seed):
        raise ValueError("split IDs must match the fixed training/validation/test IDs")
    return {name: list(split_ids[name]) for name in FIXED_SPLIT_IDS}


def fit_training_mean(
    records: Sequence[Mapping[str, object]],
    split_ids: Mapping[str, object] | None = None,
    seed: int = FIXED_SEED,
) -> dict:
    validated = validate_records([dict(record) for record in records])
    if len(validated) != 6:
        raise ValueError("baseline fitting requires the six-row fixture")
    if split_ids is None:
        split_ids = fixed_split_ids(seed)
    checked_splits = _validate_split_ids(dict(split_ids), seed)
    by_id = {record["record_id"]: record for record in validated}
    if set(by_id) != set(sum(checked_splits.values(), [])):
        raise ValueError("split IDs must cover the fixture exactly")
    training = [by_id[record_id] for record_id in checked_splits["train"]]
    training_mean = float(
        sum(float(record["target"]) for record in training) / len(training)
    )
    return {
        "schema": BASELINE_SCHEMA,
        "seed": seed,
        "training_ids": list(checked_splits["train"]),
        "training_mean": training_mean,
    }


def predict(
    baseline: Mapping[str, object], records: Sequence[Mapping[str, object]]
) -> dict[str, float]:
    if not isinstance(baseline, Mapping) or set(baseline) != {
        "schema", "seed", "training_ids", "training_mean"
    }:
        raise ValueError("baseline schema or fields are invalid")
    if baseline["schema"] != BASELINE_SCHEMA:
        raise ValueError("unsupported baseline schema")
    if baseline["seed"] != FIXED_SEED or baseline["training_ids"] != FIXED_SPLIT_IDS["train"]:
        raise ValueError("baseline must use the fixed training partition")
    training_mean = baseline["training_mean"]
    if isinstance(training_mean, bool) or not isinstance(training_mean, (int, float)):
        raise ValueError("training_mean must be a finite number")
    if not math.isfinite(training_mean):
        raise ValueError("training_mean must be a finite number")
    validated = validate_records([dict(record) for record in records])
    return {record["record_id"]: float(training_mean) for record in validated}


def mean_absolute_error(
    records: Sequence[Mapping[str, object]], predictions: Mapping[str, float]
) -> float:
    validated = validate_records([dict(record) for record in records])
    expected_ids = {record["record_id"] for record in validated}
    if not isinstance(predictions, Mapping) or set(predictions) != expected_ids:
        raise ValueError("prediction IDs must exactly match evaluation record IDs")
    errors = []
    for record in validated:
        prediction = predictions[record["record_id"]]
        if isinstance(prediction, bool) or not isinstance(prediction, (int, float)):
            raise ValueError("predictions must be finite numbers")
        if not math.isfinite(prediction):
            raise ValueError("predictions must be finite numbers")
        errors.append(abs(record["target"] - prediction))
    if not errors:
        raise ValueError("MAE requires at least one evaluation record")
    return float(sum(errors) / len(errors))


def run_baseline(records: Sequence[Mapping[str, object]]) -> dict:
    validated = validate_records([dict(record) for record in records])
    if len(validated) != 6:
        raise ValueError("baseline evaluation requires the six-row fixture")
    split_ids = fixed_split_ids()
    baseline = fit_training_mean(validated, split_ids=split_ids)
    by_id = {record["record_id"]: record for record in validated}
    validation = [by_id[record_id] for record_id in split_ids["validation"]]
    test = [by_id[record_id] for record_id in split_ids["test"]]
    validation_predictions = predict(baseline, validation)
    test_predictions = predict(baseline, test)
    held_out = validation + test
    held_out_predictions = {
        **validation_predictions,
        **test_predictions,
    }
    return {
        "split_ids": split_ids,
        "baseline": baseline,
        "validation_predictions": validation_predictions,
        "test_predictions": test_predictions,
        "validation_mae": mean_absolute_error(validation, validation_predictions),
        "test_mae": mean_absolute_error(test, test_predictions),
        "held_out_mae": mean_absolute_error(held_out, held_out_predictions),
    }
