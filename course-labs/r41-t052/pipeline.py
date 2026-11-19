"""Deterministic, leakage-checked split manifest for R41-T052."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Mapping, Sequence


FIXTURE_SCHEMA = "r41-t052.fixture.v1"
MANIFEST_SCHEMA = "r41-t052.split.v1"
FIXED_SEED = 42
REQUIRED_FIELDS = {"record_id", "x1", "x2", "target"}
OPTIONAL_FIELDS = {"group_id", "user_id", "timestamp"}
FIXED_SPLIT_IDS = {
    "train": ["r1", "r2", "r3", "r4"],
    "validation": ["r5"],
    "test": ["r6"],
}
SPLIT_NAMES = ("train", "validation", "test")


def validate_records(records: object) -> list[dict]:
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    if len(records) > 32:
        raise ValueError("records must contain at most 32 samples")

    validated = []
    seen_ids = set()
    observed_keys = None
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("record at index %d must be an object" % index)
        keys = set(record)
        if not REQUIRED_FIELDS <= keys or not keys <= REQUIRED_FIELDS | OPTIONAL_FIELDS:
            raise ValueError(
                "record at index %d must contain the required fields and only supported metadata"
                % index
            )
        if observed_keys is None:
            observed_keys = keys
        elif keys != observed_keys:
            raise ValueError("all records must use the same schema fields")

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
        for field in ("group_id", "user_id"):
            if field in record:
                value = record[field]
                if not isinstance(value, str) or not value:
                    raise ValueError("%s must be a non-empty string" % field)
                checked[field] = value
        if "timestamp" in record:
            timestamp = record["timestamp"]
            if not isinstance(timestamp, str):
                raise ValueError("timestamp must be an ISO date string")
            try:
                parsed_date = date.fromisoformat(timestamp)
            except ValueError as error:
                raise ValueError("timestamp must be an ISO date string") from error
            if parsed_date.isoformat() != timestamp:
                raise ValueError("timestamp must use YYYY-MM-DD format")
            checked["timestamp"] = timestamp
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
        raise ValueError("the R41-T052 fixture must contain exactly six rows")
    if [record["record_id"] for record in records] != [
        "r1", "r2", "r3", "r4", "r5", "r6"
    ]:
        raise ValueError("fixture IDs must be ordered r1 through r6")
    if any(record["x2"] != 1 for record in records):
        raise ValueError("fixture x2 values must all equal 1")
    if any(record["target"] != 2 * record["x1"] + 1 for record in records):
        raise ValueError("fixture targets must follow target = 2*x1 + 1")
    return records


def _validate_isolation(
    records: Sequence[Mapping[str, object]], split_ids: Mapping[str, Sequence[str]]
) -> None:
    record_by_id = {record["record_id"]: record for record in records}
    split_by_id = {
        record_id: split_name
        for split_name in SPLIT_NAMES
        for record_id in split_ids[split_name]
    }

    for field in ("group_id", "user_id"):
        if field not in records[0]:
            continue
        split_by_value = {}
        for record in records:
            value = record[field]
            split_by_value.setdefault(value, set()).add(split_by_id[record["record_id"]])
        if any(len(partitions) > 1 for partitions in split_by_value.values()):
            raise ValueError("%s values must not cross split boundaries" % field)

    if "timestamp" in records[0]:
        latest_previous_split = None
        for split_name in SPLIT_NAMES:
            timestamps = [
                date.fromisoformat(record_by_id[record_id]["timestamp"])
                for record_id in split_ids[split_name]
            ]
            earliest = min(timestamps)
            latest = max(timestamps)
            if latest_previous_split is not None and earliest <= latest_previous_split:
                raise ValueError("time-based splits must be strictly chronological")
            latest_previous_split = latest


def validate_manifest(records: Sequence[Mapping[str, object]], manifest: object) -> dict:
    validated = validate_records([dict(record) for record in records])
    if len(validated) != 6:
        raise ValueError("manifest validation requires the six-row fixture")
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "seed", "split_ids"
    }:
        raise ValueError("manifest contains unknown or missing fields")
    if manifest["schema_version"] != MANIFEST_SCHEMA:
        raise ValueError("unsupported manifest schema version")
    if type(manifest["seed"]) is not int or manifest["seed"] != FIXED_SEED:
        raise ValueError("manifest must record the fixed seed 42")

    split_ids = manifest["split_ids"]
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
        raise ValueError("record IDs must not overlap between splits")
    if set(flattened) != {record["record_id"] for record in validated}:
        raise ValueError("split IDs must cover all fixture records exactly")
    if split_ids != FIXED_SPLIT_IDS:
        raise ValueError("split IDs must match the specified r1-r4/r5/r6 partition")

    _validate_isolation(validated, split_ids)
    return {
        "schema_version": MANIFEST_SCHEMA,
        "seed": FIXED_SEED,
        "split_ids": {name: list(split_ids[name]) for name in SPLIT_NAMES},
    }


def create_manifest(
    records: Sequence[Mapping[str, object]], seed: int = FIXED_SEED
) -> dict:
    return validate_manifest(
        records,
        {
            "schema_version": MANIFEST_SCHEMA,
            "seed": seed,
            "split_ids": FIXED_SPLIT_IDS,
        },
    )


def load_manifest(path: str | Path, records: Sequence[Mapping[str, object]]) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return validate_manifest(records, manifest)
