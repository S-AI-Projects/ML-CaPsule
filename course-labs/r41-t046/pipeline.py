"""Bounded deterministic classifier smoke test for the R41-T046 fixture."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random


FEATURE_ORDER = ("x1", "x2")
ROW_FIELDS = frozenset(("record_id", "group_id", "x1", "x2", "label"))
LABELS = (0, 1)
SEED = 46
MAX_SAMPLES = 32


def validate_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")
    if len(records) > MAX_SAMPLES:
        raise ValueError(f"records must not exceed {MAX_SAMPLES} samples")

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
        raise ValueError("the fixed smoke-test fixture must contain exactly eight rows")
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
    split_names = ("train", "validation", "test")
    if not isinstance(split, dict) or set(split) != set(split_names):
        raise ValueError(f"split must contain exactly {list(split_names)}")

    by_id = {row["record_id"]: row for row in records}
    assigned = {}
    for split_name in split_names:
        ids = split[split_name]
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{split_name} must be a non-empty list of record IDs")
        for record_id in ids:
            if record_id not in by_id:
                raise ValueError(f"unknown record_id in {split_name}: {record_id}")
            if record_id in assigned:
                raise ValueError(f"record_id appears in multiple splits: {record_id}")
            assigned[record_id] = split_name

    if set(assigned) != set(by_id):
        raise ValueError("split must assign every record exactly once")

    group_splits = {}
    for record_id, split_name in assigned.items():
        group_id = by_id[record_id]["group_id"]
        previous_split = group_splits.setdefault(group_id, split_name)
        if previous_split != split_name:
            raise ValueError(f"group_id appears in multiple splits: {group_id}")

    expected_counts = {"train": 2, "validation": 1, "test": 1}
    for split_name, per_label_count in expected_counts.items():
        ids = split[split_name]
        counts = {label: sum(by_id[item]["label"] == label for item in ids) for label in LABELS}
        if len(ids) != per_label_count * len(LABELS) or counts != {
            label: per_label_count for label in LABELS
        }:
            raise ValueError(f"{split_name} is not balanced across labels 0 and 1")


@dataclass(frozen=True)
class MajorityBaseline:
    positive_probability: float
    predicted_label: int
    update_steps: int = 0

    def predict(self, records):
        validate_records(records)
        probabilities = [self.positive_probability] * len(records)
        predictions = [self.predicted_label] * len(records)
        return {
            "probabilities": probabilities,
            "predictions": predictions,
            "probability_shape": [len(probabilities)],
            "prediction_shape": [len(predictions)],
        }


def fit_majority_baseline(training_records):
    """Fit the explicitly naive constant baseline from training labels only."""
    validate_records(training_records)
    counts = {label: sum(row["label"] == label for row in training_records) for label in LABELS}
    if any(counts[label] == 0 for label in LABELS):
        raise ValueError("training records must contain both labels 0 and 1")
    majority = min(label for label in LABELS if counts[label] == max(counts.values()))
    return MajorityBaseline(
        positive_probability=counts[1] / len(training_records),
        predicted_label=majority,
    )


def evaluate(model, records, training_ids):
    validate_records(records)
    overlap = {row["record_id"] for row in records}.intersection(training_ids)
    if overlap:
        raise ValueError(f"evaluation records overlap training: {sorted(overlap)}")

    result = model.predict(records)
    epsilon = 1e-15
    losses = []
    for row, probability in zip(records, result["probabilities"]):
        bounded_probability = min(max(probability, epsilon), 1.0 - epsilon)
        losses.append(
            -(
                row["label"] * math.log(bounded_probability)
                + (1 - row["label"]) * math.log(1.0 - bounded_probability)
            )
        )
    result.update(
        {
            "record_ids": [row["record_id"] for row in records],
            "y_true": [row["label"] for row in records],
            "loss": math.fsum(losses) / len(losses),
        }
    )
    return result


def run_pipeline(fixture):
    if not isinstance(fixture, dict) or fixture.get("data_kind") != "simulated_pipeline_fixture":
        raise ValueError("fixture must be explicitly marked as simulated")
    records = fixture.get("records")
    validate_fixture(records)
    seed = fixture.get("seed")
    if type(seed) is not int:
        raise ValueError("fixture seed must be an integer")

    split = create_stratified_split(records, seed)
    rows_by_id = {row["record_id"]: row for row in records}
    training = [rows_by_id[record_id] for record_id in split["train"]]
    training_ids = {row["record_id"] for row in training}
    model = fit_majority_baseline(training)
    outputs = {
        split_name: evaluate(
            model,
            [rows_by_id[record_id] for record_id in record_ids],
            training_ids if split_name != "train" else set(),
        )
        for split_name, record_ids in split.items()
    }
    return {
        "data_kind": fixture["data_kind"],
        "seed": seed,
        "model": "training_majority_constant_baseline",
        "update_steps": model.update_steps,
        "split_record_ids": split,
        "outputs": outputs,
    }


def load_fixture():
    fixture_path = Path(__file__).with_name("fixture.json")
    with fixture_path.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


if __name__ == "__main__":
    print(json.dumps(run_pipeline(load_fixture()), indent=2))
