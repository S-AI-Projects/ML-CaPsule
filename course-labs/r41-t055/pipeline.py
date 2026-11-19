"""Training-only standardization contract for the R41-T055 fixture."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Mapping, Sequence


FIXTURE_SCHEMA = "r41-t055.fixture.v1"
SCALER_SCHEMA = "r41-t055.scaler.v1"
FEATURE_ORDER = ("x1", "x2")
SPLIT_NAMES = ("train", "validation", "test")


def validate_records(records: object) -> list[dict]:
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    if len(records) > 32:
        raise ValueError("records must contain at most 32 points")

    validated = []
    seen_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("record at index %d must be an object" % index)
        if set(record) != {"record_id", "x1", "x2"}:
            raise ValueError(
                "record at index %d must contain exactly record_id, x1, x2" % index
            )
        record_id = record["record_id"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record_id must be a non-empty string")
        if record_id in seen_ids:
            raise ValueError("duplicate record_id: %s" % record_id)

        point = {"record_id": record_id}
        for feature in FEATURE_ORDER:
            value = record[feature]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("%s must be a finite number" % feature)
            if not math.isfinite(value):
                raise ValueError("%s must be a finite number" % feature)
            point[feature] = value
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
    if len(records) != 6:
        raise ValueError("the R41-T055 fixture must contain exactly six points")
    return records


def fixed_split_ids(records: Sequence[Mapping[str, object]], seed: int = 42) -> dict:
    validated = validate_records([dict(record) for record in records])
    if len(validated) != 6:
        raise ValueError("the fixed split requires exactly six points")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    record_ids = [record["record_id"] for record in validated]
    random.Random(seed).shuffle(record_ids)
    return {
        "train": record_ids[:4],
        "validation": record_ids[4:5],
        "test": record_ids[5:],
    }


def _validate_split_ids(
    split_ids: object, records: Sequence[Mapping[str, object]], seed: int
) -> dict:
    if not isinstance(split_ids, dict) or set(split_ids) != set(SPLIT_NAMES):
        raise ValueError("split_ids must define train, validation, and test")
    if any(not isinstance(split_ids[name], list) for name in SPLIT_NAMES):
        raise ValueError("each split must be a list of record IDs")
    flattened = [
        record_id
        for split_name in SPLIT_NAMES
        for record_id in split_ids[split_name]
    ]
    if any(not isinstance(record_id, str) or not record_id for record_id in flattened):
        raise ValueError("split IDs must be non-empty strings")
    if len(flattened) != len(set(flattened)):
        raise ValueError("split IDs overlap; held-out contamination is not allowed")
    if split_ids != fixed_split_ids(records, seed=seed):
        raise ValueError("split IDs must match the fixed seed partition")
    return {name: list(split_ids[name]) for name in SPLIT_NAMES}


def fit_scaler(
    records: Sequence[Mapping[str, object]],
    split_ids: Mapping[str, object] | None = None,
    seed: int = 42,
) -> dict:
    """Fit population means/scales using only the canonical training IDs."""
    validated = validate_records([dict(record) for record in records])
    if len(validated) != 6:
        raise ValueError("scaler fitting requires the six-point fixture")
    canonical_splits = fixed_split_ids(validated, seed=seed)
    if split_ids is None:
        split_ids = canonical_splits
    checked_splits = _validate_split_ids(split_ids, validated, seed)
    by_id = {record["record_id"]: record for record in validated}
    training = [by_id[record_id] for record_id in checked_splits["train"]]

    means = []
    scales = []
    for feature in FEATURE_ORDER:
        values = [float(record[feature]) for record in training]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        scale = math.sqrt(variance)
        means.append(float(mean))
        scales.append(float(scale) if scale > 0 else 1.0)
    return {
        "schema": SCALER_SCHEMA,
        "seed": seed,
        "feature_order": list(FEATURE_ORDER),
        "training_ids": list(checked_splits["train"]),
        "mean": means,
        "scale": scales,
        "dtype": "float64",
    }


def validate_scaler(scaler: object) -> dict:
    required = {
        "schema",
        "seed",
        "feature_order",
        "training_ids",
        "mean",
        "scale",
        "dtype",
    }
    if not isinstance(scaler, dict) or set(scaler) != required:
        raise ValueError("scaler schema or fields are invalid")
    if scaler["schema"] != SCALER_SCHEMA:
        raise ValueError("unsupported scaler schema")
    if type(scaler["seed"]) is not int or scaler["seed"] < 0:
        raise ValueError("scaler seed must be a non-negative integer")
    if scaler["feature_order"] != list(FEATURE_ORDER):
        raise ValueError("scaler feature_order is incompatible")
    if scaler["dtype"] != "float64":
        raise ValueError("scaler dtype must be float64")
    if not isinstance(scaler["training_ids"], list) or not scaler["training_ids"]:
        raise ValueError("scaler training_ids must be a non-empty list")
    if len(scaler["training_ids"]) != len(set(scaler["training_ids"])):
        raise ValueError("scaler training_ids must be unique")
    for field in ("mean", "scale"):
        values = scaler[field]
        if not isinstance(values, list) or len(values) != len(FEATURE_ORDER):
            raise ValueError("%s must match feature_order shape" % field)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("%s values must be finite numbers" % field)
    if any(value <= 0 for value in scaler["scale"]):
        raise ValueError("scale values must be positive")
    return scaler


def save_scaler(scaler: Mapping[str, object], path: str | Path) -> None:
    validated = validate_scaler(dict(scaler))
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(validated, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_scaler(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        scaler = json.load(handle)
    return validate_scaler(scaler)


def transform(
    scaler: Mapping[str, object], records: Sequence[Mapping[str, object]]
) -> dict:
    """Transform records with retained parameters; this function never fits."""
    checked_scaler = validate_scaler(dict(scaler))
    validated = validate_records([dict(record) for record in records])
    means = checked_scaler["mean"]
    scales = checked_scaler["scale"]
    values = [
        [
            float((record[feature] - means[index]) / scales[index])
            for index, feature in enumerate(FEATURE_ORDER)
        ]
        for record in validated
    ]
    return {
        "record_ids": [record["record_id"] for record in validated],
        "values": values,
        "dtype": checked_scaler["dtype"],
        "shape": [len(values), len(FEATURE_ORDER)],
    }


def nearest_centroid_assignments(
    records: Sequence[Mapping[str, object]],
    centroids: Mapping[int, Sequence[float]] = {0: (0, 0), 1: (5, 5)},
) -> dict[str, int]:
    validated = validate_records([dict(record) for record in records])
    if set(centroids) != {0, 1}:
        raise ValueError("centroids must be keyed by cluster IDs 0 and 1")
    assignments = {}
    for record in validated:
        distances = {
            cluster_id: (record["x1"] - center[0]) ** 2
            + (record["x2"] - center[1]) ** 2
            for cluster_id, center in centroids.items()
        }
        assignments[record["record_id"]] = min(
            distances, key=lambda cluster_id: (distances[cluster_id], cluster_id)
        )
    return assignments
