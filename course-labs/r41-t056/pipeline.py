"""Bounded standard-library decision-stump smoke test for R41-T056."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Mapping, Sequence


FIXTURE_SCHEMA = "r41-t056.fixture.v1"
MAX_SAMPLES = 32
MAX_STEPS = 20
FEATURE_ORDER = ("x1", "x2")


def validate_records(records: object) -> list[dict]:
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    if len(records) > MAX_SAMPLES:
        raise ValueError("records must contain at most %d samples" % MAX_SAMPLES)

    validated = []
    seen_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("record at index %d must be an object" % index)
        if set(record) != {"record_id", "group_id", "x1", "x2", "label"}:
            raise ValueError(
                "record at index %d must contain exactly record_id, group_id, x1, x2, label"
                % index
            )
        record_id = record["record_id"]
        group_id = record["group_id"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        if not isinstance(group_id, str) or not group_id:
            raise ValueError("group_id must be a non-empty string")
        if record_id in seen_ids:
            raise ValueError("duplicate record_id: %s" % record_id)

        point = {"record_id": record_id, "group_id": group_id}
        for feature in FEATURE_ORDER:
            value = record[feature]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("%s must be a finite number" % feature)
            if not math.isfinite(value):
                raise ValueError("%s must be a finite number" % feature)
            point[feature] = value
        label = record["label"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError("label must be the integer 0 or 1")
        point["label"] = label
        seen_ids.add(record_id)
        validated.append(point)
    return validated


def load_fixture(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("fixture schema mismatch: expected %s" % FIXTURE_SCHEMA)
    if payload.get("simulated") is not True:
        raise ValueError("fixture must be explicitly marked simulated")
    records = validate_records(payload.get("records"))
    if len(records) != 8:
        raise ValueError("the R41-T056 fixture must contain exactly eight rows")
    return records


def fixed_stratified_splits(
    records: Sequence[Mapping[str, object]], seed: int = 42
) -> dict[str, list[dict]]:
    """Split balanced groups 4/2/2 without separating group members."""
    validated = validate_records([dict(record) for record in records])
    if len(validated) != 8:
        raise ValueError("the fixed split requires exactly eight records")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    grouped = {}
    for record in validated:
        grouped.setdefault(record["group_id"], []).append(record)
    if len(grouped) != 4 or any(
        len(group) != 2 or {row["label"] for row in group} != {0, 1}
        for group in grouped.values()
    ):
        raise ValueError("each of four groups must contain one row of each label")

    group_ids = sorted(grouped)
    random.Random(seed).shuffle(group_ids)
    assigned_groups = {
        "train": set(group_ids[:2]),
        "validation": {group_ids[2]},
        "test": {group_ids[3]},
    }
    return {
        split_name: [
            record
            for record in validated
            if record["group_id"] in assigned_groups[split_name]
        ]
        for split_name in ("train", "validation", "test")
    }


def _majority_label(labels: Sequence[int]) -> int:
    counts = {label: labels.count(label) for label in set(labels)}
    return min(counts, key=lambda label: (-counts[label], label))


def fit_stump(
    records: Sequence[Mapping[str, object]],
    seed: int = 42,
    max_steps: int = MAX_STEPS,
) -> dict:
    """Fit one decision-stump pass, capped at 20 threshold evaluations."""
    training = validate_records([dict(record) for record in records])
    if not training:
        raise ValueError("training records must not be empty")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if type(max_steps) is not int or not 1 <= max_steps <= MAX_STEPS:
        raise ValueError("max_steps must be between 1 and %d" % MAX_STEPS)
    if {record["label"] for record in training} != {0, 1}:
        raise ValueError("training records must contain both labels")

    best = None
    step_count = 0
    for feature in FEATURE_ORDER:
        values = sorted({record[feature] for record in training})
        thresholds = [(left + right) / 2 for left, right in zip(values, values[1:])]
        for threshold in thresholds:
            if step_count >= max_steps:
                break
            left = [record["label"] for record in training if record[feature] <= threshold]
            right = [record["label"] for record in training if record[feature] > threshold]
            if not left or not right:
                continue
            left_label = _majority_label(left)
            right_label = _majority_label(right)
            errors = sum(label != left_label for label in left) + sum(
                label != right_label for label in right
            )
            candidate = (errors, FEATURE_ORDER.index(feature), threshold)
            if best is None or candidate < best:
                best = candidate
                best_model = {
                    "feature_order": list(FEATURE_ORDER),
                    "split_feature": feature,
                    "threshold": threshold,
                    "left_label": left_label,
                    "right_label": right_label,
                }
            step_count += 1
        if step_count >= max_steps:
            break
    if best is None:
        raise ValueError("no valid decision-stump split was found")

    training_predictions = [
        best_model["left_label"]
        if record[best_model["split_feature"]] <= best_model["threshold"]
        else best_model["right_label"]
        for record in training
    ]
    training_loss = sum(
        prediction != record["label"]
        for prediction, record in zip(training_predictions, training)
    ) / len(training)
    return {
        "schema": "r41-t056.stump.v1",
        **best_model,
        "training_loss": training_loss,
        "training_epochs": 1,
        "step_count": step_count,
        "seed": seed,
    }


def predict(model: Mapping[str, object], records: Sequence[Mapping[str, object]]) -> list[int]:
    if not isinstance(model, Mapping):
        raise ValueError("model must be a mapping")
    required = {
        "schema",
        "feature_order",
        "split_feature",
        "threshold",
        "left_label",
        "right_label",
        "training_loss",
        "training_epochs",
        "step_count",
        "seed",
    }
    if set(model) != required or model.get("schema") != "r41-t056.stump.v1":
        raise ValueError("model schema or fields are invalid")
    if model["feature_order"] != list(FEATURE_ORDER):
        raise ValueError("model feature_order is incompatible")
    if model["split_feature"] not in FEATURE_ORDER:
        raise ValueError("model split_feature is incompatible")
    if isinstance(model["threshold"], bool) or not isinstance(
        model["threshold"], (int, float)
    ) or not math.isfinite(model["threshold"]):
        raise ValueError("model threshold must be finite")
    for field in ("left_label", "right_label"):
        if type(model[field]) is not int or model[field] not in (0, 1):
            raise ValueError("model labels must be integers 0 or 1")

    validated = validate_records([dict(record) for record in records])
    feature = model["split_feature"]
    return [
        model["left_label"]
        if record[feature] <= model["threshold"]
        else model["right_label"]
        for record in validated
    ]


def run_smoke(records: Sequence[Mapping[str, object]], seed: int = 42) -> dict:
    validated = validate_records([dict(record) for record in records])
    splits = fixed_stratified_splits(validated, seed=seed)
    model = fit_stump(splits["train"], seed=seed, max_steps=MAX_STEPS)
    targets = splits["validation"] + splits["test"]
    predictions = predict(model, targets)
    prediction_map = {
        record["record_id"]: prediction
        for record, prediction in zip(targets, predictions)
    }
    loss = sum(
        prediction != record["label"]
        for prediction, record in zip(predictions, targets)
    ) / len(targets)
    return {
        "split_ids": {
            split_name: [record["record_id"] for record in splits[split_name]]
            for split_name in ("train", "validation", "test")
        },
        "model": model,
        "prediction_record_ids": [record["record_id"] for record in targets],
        "predictions": prediction_map,
        "prediction_shape": [len(predictions)],
        "loss": loss,
        "step_count": model["step_count"],
    }
