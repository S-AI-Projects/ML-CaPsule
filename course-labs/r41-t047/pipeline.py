"""Reproducible feature-selection classification for the R41-T047 fixture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any


RECORD_FIELDS = {"record_id", "group_id", "x1", "x2", "label"}
FEATURE_FIELDS = ("x1", "x2")
SPLIT_NAMES = ("train", "validation", "test")
ALGORITHM_FIELDS = {
    "name",
    "selector",
    "n_features",
    "distance",
    "tie_break",
}


class ConfigurationError(ValueError):
    """Raised when fixture data or serialized configuration violates its contract."""


def validate_records(records: Sequence[Mapping[str, Any]]) -> None:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence) or not records:
        raise ConfigurationError("records must be a non-empty sequence")

    record_ids: set[str] = set()
    group_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != RECORD_FIELDS:
            raise ConfigurationError(
                f"record {index} must contain exactly {sorted(RECORD_FIELDS)}"
            )
        for field in ("record_id", "group_id"):
            value = record[field]
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError(f"record {index} has invalid {field}")
        if record["record_id"] in record_ids:
            raise ConfigurationError(f"duplicate record_id: {record['record_id']}")
        if record["group_id"] in group_ids:
            raise ConfigurationError(f"duplicate group_id: {record['group_id']}")
        record_ids.add(record["record_id"])
        group_ids.add(record["group_id"])

        for feature in FEATURE_FIELDS:
            value = record[feature]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ConfigurationError(f"record {record['record_id']} has invalid {feature}")
            if not math.isfinite(value):
                raise ConfigurationError(f"record {record['record_id']} has non-finite {feature}")
        label = record["label"]
        if type(label) is not int or label not in (0, 1):
            raise ConfigurationError(f"record {record['record_id']} label must be integer 0 or 1")


def validate_config(
    config: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    validate_records(records)
    if not isinstance(config, Mapping):
        raise ConfigurationError("config must be an object")

    required_fields = {
        "schema_version",
        "seed",
        "feature_order",
        "split_ids",
        "algorithm",
    }
    if set(config) != required_fields:
        raise ConfigurationError(
            f"config must contain exactly {sorted(required_fields)}"
        )
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ConfigurationError("schema_version must be integer 1")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ConfigurationError("seed must be a non-negative integer")

    feature_order = config["feature_order"]
    if (
        not isinstance(feature_order, list)
        or any(not isinstance(feature, str) for feature in feature_order)
        or len(feature_order) != len(FEATURE_FIELDS)
        or set(feature_order) != set(FEATURE_FIELDS)
    ):
        raise ConfigurationError("feature_order must contain x1 and x2 exactly once")

    splits = config["split_ids"]
    if not isinstance(splits, Mapping) or set(splits) != set(SPLIT_NAMES):
        raise ConfigurationError("split_ids must define train, validation, and test")
    all_ids = {record["record_id"] for record in records}
    assigned: list[str] = []
    records_by_id = {record["record_id"]: record for record in records}
    group_to_split: dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        ids = splits[split_name]
        if not isinstance(ids, list) or not ids or any(
            not isinstance(record_id, str) for record_id in ids
        ):
            raise ConfigurationError(f"{split_name} IDs must be a non-empty string list")
        if len(ids) != len(set(ids)):
            raise ConfigurationError(f"{split_name} contains duplicate IDs")
        labels: list[int] = []
        for record_id in ids:
            if record_id not in records_by_id:
                raise ConfigurationError(f"unknown record_id in {split_name}: {record_id}")
            record = records_by_id[record_id]
            labels.append(record["label"])
            group_id = record["group_id"]
            previous_split = group_to_split.setdefault(group_id, split_name)
            if previous_split != split_name:
                raise ConfigurationError(f"group_id {group_id} appears in multiple splits")
        if set(labels) != {0, 1} or abs(labels.count(0) - labels.count(1)) > 1:
            raise ConfigurationError(f"{split_name} must be stratified across both labels")
        assigned.extend(ids)

    if len(assigned) != len(set(assigned)) or set(assigned) != all_ids:
        raise ConfigurationError("split IDs must partition every input record exactly once")

    algorithm = config["algorithm"]
    if not isinstance(algorithm, Mapping) or set(algorithm) != ALGORITHM_FIELDS:
        raise ConfigurationError(
            f"algorithm must contain exactly {sorted(ALGORITHM_FIELDS)}"
        )
    expected_algorithm = {
        "name": "nearest_centroid",
        "selector": "max_class_mean_gap",
        "distance": "euclidean",
        "tie_break": "lower_label",
    }
    for field, expected in expected_algorithm.items():
        if algorithm[field] != expected:
            raise ConfigurationError(f"unsupported or contradictory algorithm setting: {field}")
    if (
        type(algorithm["n_features"]) is not int
        or not 1 <= algorithm["n_features"] <= len(feature_order)
    ):
        raise ConfigurationError("n_features must be between 1 and feature_order length")


def _feature_vector(
    values: Sequence[float], feature_order: Sequence[str]
) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ConfigurationError("features must be a sequence matching feature_order")
    if len(values) != len(feature_order):
        raise ConfigurationError(
            f"feature vector has length {len(values)}; expected {len(feature_order)}"
        )
    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigurationError("feature values must be finite numbers")
        if not math.isfinite(value):
            raise ConfigurationError("feature values must be finite numbers")
        normalized.append(float(value))
    return normalized


def _fit(
    training: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> tuple[list[str], dict[str, list[float]]]:
    feature_order = config["feature_order"]
    labels = (0, 1)
    gaps: dict[str, float] = {}
    for feature in feature_order:
        class_means = {
            label: sum(row[feature] for row in training if row["label"] == label)
            / sum(row["label"] == label for row in training)
            for label in labels
        }
        gaps[feature] = abs(class_means[1] - class_means[0])

    selected = sorted(
        feature_order,
        key=lambda feature: (-gaps[feature], feature_order.index(feature)),
    )[: config["algorithm"]["n_features"]]
    centroids = {
        str(label): [
            sum(row[feature] for row in training if row["label"] == label)
            / sum(row["label"] == label for row in training)
            for feature in selected
        ]
        for label in labels
    }
    return selected, centroids


def predict(
    values: Sequence[float],
    *,
    feature_order: Sequence[str],
    selected_features: Sequence[str],
    centroids: Mapping[str, Sequence[float]],
) -> int:
    vector = _feature_vector(values, feature_order)
    if (
        not selected_features
        or len(selected_features) != len(set(selected_features))
        or any(feature not in feature_order for feature in selected_features)
    ):
        raise ConfigurationError("selected_features must be unique members of feature_order")
    if not isinstance(centroids, Mapping) or set(centroids) != {"0", "1"}:
        raise ConfigurationError("centroids must provide labels 0 and 1")
    if any(
        not isinstance(centroid, Sequence)
        or isinstance(centroid, (str, bytes))
        or len(centroid) != len(selected_features)
        for centroid in centroids.values()
    ):
        raise ConfigurationError("centroid shape must match selected_features")
    selected_values = [
        vector[feature_order.index(feature)] for feature in selected_features
    ]
    distances = {
        int(label): sum(
            (value - center) ** 2 for value, center in zip(selected_values, centroid)
        )
        for label, centroid in centroids.items()
    }
    return min(distances, key=lambda label: (distances[label], label))


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    feature_order: Sequence[str],
    selected_features: Sequence[str],
    centroids: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    predictions = [
        predict(
            [row[feature] for feature in feature_order],
            feature_order=feature_order,
            selected_features=selected_features,
            centroids=centroids,
        )
        for row in rows
    ]
    labels = [row["label"] for row in rows]
    return {
        "record_ids": [row["record_id"] for row in rows],
        "y_true": labels,
        "y_pred": predictions,
        "accuracy": sum(actual == predicted for actual, predicted in zip(labels, predictions))
        / len(rows),
    }


def run_pipeline(
    records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    validate_config(config, records)
    records_by_id = {record["record_id"]: record for record in records}
    splits = {
        name: [records_by_id[record_id] for record_id in config["split_ids"][name]]
        for name in SPLIT_NAMES
    }
    selected_features, centroids = _fit(splits["train"], config)
    return {
        "data_kind": "simulated_pipeline_verification",
        "schema_version": config["schema_version"],
        "seed": config["seed"],
        "feature_order": config["feature_order"],
        "selected_features": selected_features,
        "algorithm": dict(config["algorithm"]),
        "centroids": centroids,
        "train_record_ids": config["split_ids"]["train"],
        "validation": _evaluate(
            splits["validation"],
            config["feature_order"],
            selected_features,
            centroids,
        ),
        "test": _evaluate(
            splits["test"],
            config["feature_order"],
            selected_features,
            centroids,
        ),
    }


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def load_fixture() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    directory = Path(__file__).parent
    fixture = load_json(directory / "fixture.json")
    if not isinstance(fixture, Mapping) or fixture.get("data_kind") != "simulated":
        raise ConfigurationError("fixture must be explicitly marked simulated")
    config = load_json(directory / "config.json")
    records = fixture.get("records")
    validate_config(config, records)
    return records, config


if __name__ == "__main__":
    records, configuration = load_fixture()
    print(json.dumps(run_pipeline(records, configuration), indent=2, sort_keys=True))
