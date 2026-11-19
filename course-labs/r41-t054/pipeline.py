"""Explicit binary classification and regression metrics for R41-T054."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Sequence


FIXTURE_SCHEMA = "r41-t054.fixture.v1"
MAX_SAMPLES = 32


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
        if set(record) != {"record_id", "text", "label"}:
            raise ValueError(
                "record at index %d must contain exactly record_id, text, label"
                % index
            )
        record_id = record["record_id"]
        text = record["text"]
        label = record["label"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        if record_id in seen_ids:
            raise ValueError("duplicate record_id: %s" % record_id)
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("label must be the integer 0 or 1")
        seen_ids.add(record_id)
        validated.append(
            {"record_id": record_id, "text": text, "label": label}
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
    if len(records) != 6:
        raise ValueError("the R41-T054 fixture must contain exactly six records")
    return records


def fixed_splits(
    records: Sequence[dict], seed: int = 42
) -> dict[str, list[dict]]:
    validated = validate_records([dict(record) for record in records])
    if len(validated) != 6:
        raise ValueError("fixed splits require exactly six records")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    shuffled = list(validated)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": shuffled[:4],
        "validation": shuffled[4:5],
        "test": shuffled[5:],
    }


def _per_class_precision_recall(
    y_true: Sequence[int], y_pred: Sequence[int], label: int
) -> tuple[float, float]:
    true_positive = sum(
        truth == label and prediction == label
        for truth, prediction in zip(y_true, y_pred)
    )
    false_positive = sum(
        truth != label and prediction == label
        for truth, prediction in zip(y_true, y_pred)
    )
    false_negative = sum(
        truth == label and prediction != label
        for truth, prediction in zip(y_true, y_pred)
    )
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator if precision_denominator else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    return float(precision), float(recall)


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    average: str = "binary",
    positive_label: int = 1,
) -> dict:
    if not isinstance(y_true, (list, tuple)) or not isinstance(y_pred, (list, tuple)):
        raise ValueError("y_true and y_pred must be lists or tuples")
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same non-zero length")
    if len(y_true) > MAX_SAMPLES:
        raise ValueError("metric inputs must contain at most 32 samples")
    if any(
        type(value) is not int or value not in (0, 1)
        for value in tuple(y_true) + tuple(y_pred)
    ):
        raise ValueError("classification labels must be integers 0 or 1")
    if average == "binary":
        if type(positive_label) is not int or positive_label not in (0, 1):
            raise ValueError("positive_label must be 0 or 1")
        precision, recall = _per_class_precision_recall(
            y_true, y_pred, positive_label
        )
        return {
            "average": "binary",
            "positive_label": positive_label,
            "zero_division": 0.0,
            "precision": precision,
            "recall": recall,
        }
    if average == "macro":
        per_class = [
            _per_class_precision_recall(y_true, y_pred, label)
            for label in (0, 1)
        ]
        return {
            "average": "macro",
            "labels": [0, 1],
            "zero_division": 0.0,
            "precision": sum(metrics[0] for metrics in per_class) / 2,
            "recall": sum(metrics[1] for metrics in per_class) / 2,
        }
    raise ValueError("average must be 'binary' or 'macro'")


def mean_absolute_error(
    y_true: Sequence[int | float], y_pred: Sequence[int | float]
) -> float:
    if not isinstance(y_true, (list, tuple)) or not isinstance(y_pred, (list, tuple)):
        raise ValueError("y_true and y_pred must be lists or tuples")
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same non-zero length")
    if len(y_true) > MAX_SAMPLES:
        raise ValueError("metric inputs must contain at most 32 samples")
    pairs = list(zip(y_true, y_pred))
    for value in (value for pair in pairs for value in pair):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("regression values must be finite numbers")
        if not math.isfinite(value):
            raise ValueError("regression values must be finite numbers")
    return float(sum(abs(truth - prediction) for truth, prediction in pairs) / len(pairs))
