"""Strict regression dataset contract and split definition for R41-T051."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Sequence


FIXTURE_SCHEMA = "r41-t051.fixture.v1"
MANIFEST_SCHEMA = "r41-t051.split.v1"
FIXED_SEED = 42
REQUIRED_FIELDS = {"record_id", "x1", "x2", "target"}
FIXED_SPLIT_IDS = {
    "train": ["r1", "r2", "r3", "r4"],
    "validation": ["r5"],
    "test": ["r6"],
}


class DatasetValidationError(ValueError):
    """Validation failure retaining every row/field diagnostic."""

    def __init__(self, errors: list[dict], input_count: int | None):
        self.errors = errors
        self.input_count = input_count
        super().__init__(
            "%d validation error(s) in %s input row(s)"
            % (len(errors), input_count if input_count is not None else "unknown")
        )

    def as_dict(self) -> dict:
        return {"input_count": self.input_count, "errors": list(self.errors)}


def _error(
    record_id: object, row_index: int | None, field: str, reason: str
) -> dict:
    return {
        "record_id": record_id if isinstance(record_id, str) else None,
        "row_index": row_index,
        "field": field,
        "reason": reason,
    }


def validate_records(records: object) -> list[dict]:
    if not isinstance(records, list):
        raise DatasetValidationError(
            [_error(None, None, "records", "expected_list")], None
        )
    if len(records) > 32:
        raise DatasetValidationError(
            [_error(None, None, "records", "exceeds_32_row_limit")], len(records)
        )

    validated = []
    errors = []
    seen_ids = set()
    for row_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            errors.append(_error(None, row_index, "record", "expected_object"))
            continue

        record_id = record.get("record_id")
        if "record_id" not in record:
            errors.append(
                _error(None, row_index, "record_id", "missing_required_field")
            )
        elif not isinstance(record_id, str) or not record_id:
            errors.append(
                _error(record_id, row_index, "record_id", "invalid_record_id")
            )
        elif record_id in seen_ids:
            errors.append(
                _error(record_id, row_index, "record_id", "duplicate_record_id")
            )
        else:
            seen_ids.add(record_id)

        for field in sorted(REQUIRED_FIELDS - {"record_id"}):
            if field not in record:
                errors.append(
                    _error(record_id, row_index, field, "missing_required_field")
                )
                continue
            value = record[field]
            if isinstance(value, bool):
                errors.append(
                    _error(record_id, row_index, field, "expected_numeric_scalar")
                )
            elif isinstance(value, (list, tuple)):
                errors.append(
                    _error(record_id, row_index, field, "tensor_or_vector_not_supported")
                )
            elif not isinstance(value, (int, float)):
                errors.append(
                    _error(record_id, row_index, field, "expected_numeric_scalar")
                )
            elif not math.isfinite(value):
                errors.append(
                    _error(record_id, row_index, field, "non_finite_number")
                )

        for field in sorted(set(record) - REQUIRED_FIELDS):
            errors.append(
                _error(record_id, row_index, field, "unexpected_field")
            )
        validated.append(dict(record))

    if errors:
        raise DatasetValidationError(errors, len(records))
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
        raise ValueError("the R41-T051 fixture must contain exactly six rows")
    if [record["record_id"] for record in records] != [
        "r1", "r2", "r3", "r4", "r5", "r6"
    ]:
        raise ValueError("fixture IDs must be ordered r1 through r6")
    if any(record["x2"] != 1 for record in records):
        raise ValueError("fixture x2 values must all equal 1")
    if any(record["target"] != 2 * record["x1"] + 1 for record in records):
        raise ValueError("fixture targets must follow target = 2*x1 + 1")
    return records


def fixed_split_manifest(seed: int = FIXED_SEED) -> dict:
    if type(seed) is not int or seed != FIXED_SEED:
        raise ValueError("the fixed split manifest uses seed 42")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "seed": FIXED_SEED,
        "split_ids": {name: list(ids) for name, ids in FIXED_SPLIT_IDS.items()},
    }


def validate_manifest(
    records: Sequence[Mapping[str, object]], manifest: object
) -> dict:
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
        raise ValueError("manifest must use seed 42")
    expected_ids = {record["record_id"] for record in validated}
    split_ids = manifest["split_ids"]
    if not isinstance(split_ids, dict) or set(split_ids) != set(FIXED_SPLIT_IDS):
        raise ValueError("split_ids must define train, validation, and test")
    if any(not isinstance(split_ids[name], list) for name in FIXED_SPLIT_IDS):
        raise ValueError("each split must be a list of record IDs")
    all_ids = [record_id for ids in split_ids.values() for record_id in ids]
    if any(not isinstance(record_id, str) for record_id in all_ids):
        raise ValueError("split IDs must be strings")
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("split record IDs must not overlap")
    if set(all_ids) != expected_ids:
        raise ValueError("split IDs must cover all fixture rows exactly")
    if split_ids != FIXED_SPLIT_IDS:
        raise ValueError("split IDs must match r1-r4/r5/r6")
    return fixed_split_manifest(seed=manifest["seed"])


def create_split_manifest(records: Sequence[Mapping[str, object]]) -> dict:
    return validate_manifest(records, fixed_split_manifest())


def split_records(
    records: Sequence[Mapping[str, object]], manifest: Mapping[str, object]
) -> dict[str, list[dict]]:
    validated = validate_records([dict(record) for record in records])
    checked_manifest = validate_manifest(validated, dict(manifest))
    by_id = {record["record_id"]: record for record in validated}
    return {
        name: [by_id[record_id] for record_id in checked_manifest["split_ids"][name]]
        for name in FIXED_SPLIT_IDS
    }
