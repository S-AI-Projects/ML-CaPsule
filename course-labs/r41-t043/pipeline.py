"""Majority-class baseline for the R41-T043 simulated text fixture."""

import json
import random
from pathlib import Path


def validate_records(records):
    if not isinstance(records, list) or not records:
        raise ValueError("records must be a non-empty list")

    seen_ids = set()
    required_fields = {"record_id", "text", "label"}
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != required_fields:
            raise ValueError(f"record {index} must contain exactly {sorted(required_fields)}")
        record_id = record["record_id"]
        text = record["text"]
        label = record["label"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"record {index} has an invalid record_id")
        if record_id in seen_ids:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen_ids.add(record_id)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"record {record_id} has invalid text")
        if type(label) is not int or label not in (0, 1):
            raise ValueError(f"record {record_id} label must be integer 0 or 1")


def split_records(records, seed):
    """Create reproducible 4/1/1 train/validation/test splits for six rows."""
    validate_records(records)
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if len(records) != 6:
        raise ValueError("the fixed 4/1/1 split requires exactly six records")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": shuffled[:4],
        "validation": shuffled[4:5],
        "test": shuffled[5:],
    }


def fit_majority_baseline(training_records):
    """Return the most frequent training label; ties go to the smaller label."""
    validate_records(training_records)
    counts = {0: 0, 1: 0}
    for record in training_records:
        counts[record["label"]] += 1
    return min((label for label in counts if counts[label] == max(counts.values())))


def evaluate_constant_label(label, records, training_ids):
    if type(label) is not int or label not in (0, 1):
        raise ValueError("prediction label must be integer 0 or 1")
    validate_records(records)
    evaluation_ids = {record["record_id"] for record in records}
    overlap = evaluation_ids.intersection(training_ids)
    if overlap:
        raise ValueError(f"evaluation records overlap training: {sorted(overlap)}")

    predictions = [label] * len(records)
    correct = sum(prediction == record["label"] for prediction, record in zip(predictions, records))
    return {
        "record_ids": [record["record_id"] for record in records],
        "y_true": [record["label"] for record in records],
        "y_pred": predictions,
        "accuracy": correct / len(records),
    }


def run_pipeline(records, seed):
    splits = split_records(records, seed)
    training_ids = {record["record_id"] for record in splits["train"]}
    prediction = fit_majority_baseline(splits["train"])
    counts = {
        str(label): sum(record["label"] == label for record in splits["train"])
        for label in (0, 1)
    }
    return {
        "seed": seed,
        "model": "training_majority_class",
        "predicted_label": prediction,
        "training_class_counts": counts,
        "train_record_ids": [record["record_id"] for record in splits["train"]],
        "validation": evaluate_constant_label(
            prediction, splits["validation"], training_ids
        ),
        "test": evaluate_constant_label(prediction, splits["test"], training_ids),
    }


def load_fixture():
    fixture_path = Path(__file__).with_name("fixture.json")
    with fixture_path.open(encoding="utf-8") as fixture_file:
        fixture = json.load(fixture_file)
    if fixture.get("data_kind") != "simulated":
        raise ValueError("fixture must be explicitly marked simulated")
    return fixture


if __name__ == "__main__":
    fixture = load_fixture()
    print(json.dumps(run_pipeline(fixture["records"], fixture["seed"]), indent=2))
