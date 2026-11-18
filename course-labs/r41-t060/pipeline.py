"""Small saved text-pipeline interface using only the Python standard library."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Mapping, Sequence


FIXTURE_SCHEMA = "r41-t060.fixture.v1"
MODEL_SCHEMA = "r41-t060.model.v1"


def validate_fixture(records: object) -> list[dict]:
    """Validate the literal fixture record contract and return a plain list."""
    if not isinstance(records, list):
        raise ValueError("fixture records must be a list")

    validated = []
    seen_ids = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("record at index %d must be an object" % index)
        if set(record) != {"record_id", "text", "label"}:
            raise ValueError(
                "record at index %d must contain exactly record_id, text, label" % index
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
        validated.append({"record_id": record_id, "text": text, "label": label})
    return validated


def load_fixture(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema") != FIXTURE_SCHEMA:
        raise ValueError("fixture schema mismatch: expected %s" % FIXTURE_SCHEMA)
    if payload.get("simulated") is not True:
        raise ValueError("fixture must be explicitly marked simulated")
    return validate_fixture(payload.get("records"))


def fixed_splits(records: Sequence[Mapping[str, object]]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return a reproducible 4/1/1 train/validation/test partition."""
    if len(records) != 6:
        raise ValueError("fixed splits require exactly six records")
    shuffled = [dict(record) for record in records]
    random.Random(42).shuffle(shuffled)
    train = shuffled[:4]
    validation = shuffled[4:5]
    test = shuffled[5:]
    return train, validation, test


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def train(records: Sequence[Mapping[str, object]], alpha: float = 1.0) -> dict:
    """Train a multinomial Naive Bayes text classifier."""
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    validated = validate_fixture([dict(record) for record in records])

    vocabulary = sorted({token for record in validated for token in tokenize(record["text"])})
    labels = sorted({record["label"] for record in validated})
    if labels != [0, 1]:
        raise ValueError("training data must contain both labels 0 and 1")

    class_counts = {label: 0 for label in labels}
    token_counts = {label: {token: 0 for token in vocabulary} for label in labels}
    for record in validated:
        label = record["label"]
        class_counts[label] += 1
        for token in tokenize(record["text"]):
            token_counts[label][token] += 1

    total = len(validated)
    class_log_prior = {
        str(label): math.log(class_counts[label] / total) for label in labels
    }
    token_log_likelihood = {}
    for label in labels:
        denominator = sum(token_counts[label].values()) + alpha * len(vocabulary)
        token_log_likelihood[str(label)] = {
            token: math.log((token_counts[label][token] + alpha) / denominator)
            for token in vocabulary
        }

    return {
        "schema": MODEL_SCHEMA,
        "alpha": alpha,
        "labels": labels,
        "vocabulary": vocabulary,
        "class_log_prior": class_log_prior,
        "token_log_likelihood": token_log_likelihood,
    }


def validate_model(model: object) -> dict:
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    if model.get("schema") != MODEL_SCHEMA:
        raise ValueError("model schema mismatch: expected %s" % MODEL_SCHEMA)
    required = {
        "schema", "alpha", "labels", "vocabulary",
        "class_log_prior", "token_log_likelihood",
    }
    if set(model) != required:
        raise ValueError("model version/parameter mismatch: unexpected model fields")
    if model["labels"] != [0, 1]:
        raise ValueError("model labels must be [0, 1]")
    if not isinstance(model["vocabulary"], list) or not all(
        isinstance(token, str) for token in model["vocabulary"]
    ):
        raise ValueError("model vocabulary must be a list of strings")
    return model


def save_model(model: Mapping[str, object], path: str | Path) -> None:
    model = validate_model(dict(model))
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(model, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_model(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        model = json.load(handle)
    return validate_model(model)


def predict(model: Mapping[str, object], texts: Sequence[str]) -> list[int]:
    model = validate_model(dict(model))
    if not isinstance(texts, (list, tuple)):
        raise ValueError("texts must be a list or tuple")
    predictions = []
    for text in texts:
        if not isinstance(text, str):
            raise ValueError("prediction text must be a string")
        scores = {}
        for label in model["labels"]:
            label_key = str(label)
            score = model["class_log_prior"][label_key]
            likelihood = model["token_log_likelihood"][label_key]
            for token in tokenize(text):
                if token in model["vocabulary"]:
                    score += likelihood[token]
            scores[label] = score
        predictions.append(max(scores, key=scores.get))
    return predictions
