"""Training-only numeric preprocessing for the R41-T045 simulated fixture."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random


FEATURE_ORDER = ("x1", "x2")
ROW_FIELDS = frozenset(("record_id", "group_id", "x1", "x2", "label"))
LABELS = (0, 1)
SEED = 45


def validate_feature_order(feature_order):
    if not isinstance(feature_order, (list, tuple)) or tuple(feature_order) != FEATURE_ORDER:
        raise ValueError(f"feature_order must be exactly {list(FEATURE_ORDER)}")


def validate_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")

    seen_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != ROW_FIELDS:
            raise ValueError(f"record {index} must contain exactly {sorted(ROW_FIELDS)}")
        record_id = record["record_id"]
        group_id = record["group_id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"record {index} has an invalid record_id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen_ids.add(record_id)
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"record {record_id} has an invalid group_id")
        for feature in FEATURE_ORDER:
            value = record[feature]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"record {record_id} has invalid or missing {feature}")
        label = record["label"]
        if type(label) is not int or label not in LABELS:
            raise ValueError(f"record {record_id} label must be integer 0 or 1")


def validate_fixture(records):
    validate_records(records)
    if len(records) != 8:
        raise ValueError("the fixed stratified fixture must contain exactly eight rows")
    counts = {label: sum(row["label"] == label for row in records) for label in LABELS}
    if counts != {0: 4, 1: 4}:
        raise ValueError("fixture label set must be exactly {0, 1}, with four rows per label")


def create_stratified_split(records, seed=SEED):
    validate_fixture(records)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    group_ids = [row["group_id"] for row in records]
    if len(set(group_ids)) != len(group_ids):
        raise ValueError("the fixed stratified split requires one row per group_id")

    randomizer = random.Random(seed)
    split = {"train": [], "validation": [], "test": []}
    for label in LABELS:
        class_ids = [row["record_id"] for row in records if row["label"] == label]
        randomizer.shuffle(class_ids)
        split["train"].extend(class_ids[:2])
        split["validation"].append(class_ids[2])
        split["test"].append(class_ids[3])
    validate_split(records, split)
    return split


def validate_split(records, split):
    validate_fixture(records)
    expected_names = ("train", "validation", "test")
    if not isinstance(split, dict) or set(split) != set(expected_names):
        raise ValueError(f"split must contain exactly {list(expected_names)}")

    by_id = {row["record_id"]: row for row in records}
    assigned = {}
    for split_name in expected_names:
        ids = split[split_name]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{split_name} must be a non-empty list of record IDs")
        for record_id in ids:
            if record_id not in by_id:
                raise ValueError(f"unknown record_id in {split_name}: {record_id}")
            if record_id in assigned:
                raise ValueError(f"record_id appears in multiple splits: {record_id}")
            assigned[record_id] = split_name

    expected_ids = set(by_id)
    if set(assigned) != expected_ids:
        raise ValueError("split must assign every record exactly once")

    group_splits = {}
    for record_id, split_name in assigned.items():
        group_id = by_id[record_id]["group_id"]
        previous_split = group_splits.setdefault(group_id, split_name)
        if previous_split != split_name:
            raise ValueError(f"group_id appears in multiple splits: {group_id}")

    expected_counts = {"train": 2, "validation": 1, "test": 1}
    for split_name, expected_count in expected_counts.items():
        ids = split[split_name]
        label_counts = {label: sum(by_id[item]["label"] == label for item in ids) for label in LABELS}
        if len(ids) != expected_count * len(LABELS) or label_counts != {
            label: expected_count for label in LABELS
        }:
            raise ValueError(f"{split_name} is not balanced across labels 0 and 1")


@dataclass(frozen=True)
class NumericPreprocessor:
    feature_order: tuple
    means: tuple
    scales: tuple

    def transform(self, records, feature_order=FEATURE_ORDER):
        validate_feature_order(feature_order)
        if tuple(feature_order) != self.feature_order:
            raise ValueError("transform feature order must match the fitted feature order")
        validate_records(records)
        values = [
            [
                (float(row[feature]) - mean) / scale
                for feature, mean, scale in zip(self.feature_order, self.means, self.scales)
            ]
            for row in records
        ]
        return {
            "values": values,
            "dtype": "float64",
            "shape": [len(values), len(self.feature_order)],
        }


def fit_preprocessor(training_records, feature_order=FEATURE_ORDER):
    """Fit means and population standard deviations from training rows only."""
    validate_feature_order(feature_order)
    validate_records(training_records)
    means = []
    scales = []
    for feature in feature_order:
        column = [float(row[feature]) for row in training_records]
        mean = math.fsum(column) / len(column)
        variance = math.fsum((value - mean) ** 2 for value in column) / len(column)
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale else 1.0)
    return NumericPreprocessor(tuple(feature_order), tuple(means), tuple(scales))


def run_pipeline(fixture):
    if not isinstance(fixture, dict) or fixture.get("data_kind") != "simulated_pipeline_fixture":
        raise ValueError("fixture must be explicitly marked as simulated")
    records = fixture.get("records")
    validate_fixture(records)
    seed = fixture.get("seed")
    if type(seed) is not int:
        raise ValueError("fixture seed must be an integer")

    split = create_stratified_split(records, seed)
    records_by_id = {row["record_id"]: row for row in records}
    train_rows = [records_by_id[item] for item in split["train"]]
    preprocessor = fit_preprocessor(train_rows, feature_order=FEATURE_ORDER)
    transformed = {
        split_name: preprocessor.transform(
            [records_by_id[item] for item in record_ids],
            feature_order=FEATURE_ORDER,
        )
        for split_name, record_ids in split.items()
    }
    return {
        "data_kind": fixture["data_kind"],
        "seed": seed,
        "feature_order": list(FEATURE_ORDER),
        "split_record_ids": split,
        "training_mean": list(preprocessor.means),
        "training_scale": list(preprocessor.scales),
        "transformed": transformed,
    }


def load_fixture():
    fixture_path = Path(__file__).with_name("fixture.json")
    with fixture_path.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


if __name__ == "__main__":
    print(json.dumps(run_pipeline(load_fixture()), indent=2))
