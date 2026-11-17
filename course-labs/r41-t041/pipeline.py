"""Schema validation for the deterministic concrete-strength fixture.

This module validates input records only. It does not train a model or claim
that the simulated values represent measured concrete strength.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = ("record_id", "x1", "x2", "target")
NUMERIC_FIELDS = ("x1", "x2", "target")
MISSING = object()


@dataclass(frozen=True)
class ValidationIssue:
    """One schema violation, tied to a record and field."""

    record_id: str
    field: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "record_id": self.record_id,
            "field": self.field,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Validation outcome; record_count includes valid and invalid rows."""

    record_count: int
    issues: tuple[ValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_count": self.record_count,
            "valid": self.is_valid,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def load_fixture(path: str | Path | None = None) -> dict[str, Any]:
    """Load the checked-in JSON fixture without coercing field values."""

    fixture_path = Path(path) if path is not None else Path(__file__).with_name(
        "fixture.json"
    )
    with fixture_path.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def _display_record_id(value: Any) -> str:
    if value is MISSING:
        return "<missing>"
    if isinstance(value, str):
        return value if value else "<empty>"
    return repr(value)


def _is_finite_real(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, TypeError, ValueError):
        return False


def validate_records(records: Iterable[Any]) -> ValidationReport:
    """Validate every row and report every issue without filtering records.

    Numeric fields must be finite Python integers or floats; booleans, numeric
    text, containers, and tensor-like objects are rejected rather than coerced.
    """

    if isinstance(records, (str, bytes, Mapping)):
        return ValidationReport(
            record_count=0,
            issues=(
                ValidationIssue(
                    record_id="<input>",
                    field="records",
                    reason="must be an iterable of record objects",
                ),
            ),
        )

    try:
        iterator = iter(records)
    except TypeError:
        return ValidationReport(
            record_count=0,
            issues=(
                ValidationIssue(
                    record_id="<input>",
                    field="records",
                    reason="must be an iterable of record objects",
                ),
            ),
        )

    issues: list[ValidationIssue] = []
    seen_ids: dict[str, int] = {}
    record_count = 0

    for row_number, row in enumerate(iterator, start=1):
        record_count += 1
        if not isinstance(row, Mapping):
            issues.append(
                ValidationIssue(
                    record_id="<unknown>",
                    field="<record>",
                    reason="must be a mapping/object",
                )
            )
            continue

        raw_record_id = row.get("record_id", MISSING)
        display_id = _display_record_id(raw_record_id)
        if raw_record_id is MISSING:
            issues.append(
                ValidationIssue(display_id, "record_id", "is required")
            )
        elif not isinstance(raw_record_id, str):
            issues.append(
                ValidationIssue(display_id, "record_id", "must be a string")
            )
        elif not raw_record_id.strip():
            issues.append(
                ValidationIssue(
                    display_id, "record_id", "must be a non-empty string"
                )
            )
        elif raw_record_id in seen_ids:
            issues.append(
                ValidationIssue(
                    display_id,
                    "record_id",
                    f"duplicates record_id first seen in row {seen_ids[raw_record_id]}",
                )
            )
        else:
            seen_ids[raw_record_id] = row_number

        for field in REQUIRED_FIELDS:
            if field not in row and field != "record_id":
                issues.append(
                    ValidationIssue(display_id, field, "is required")
                )

        for field in NUMERIC_FIELDS:
            if field not in row:
                continue
            if not isinstance(row[field], (int, float)) or isinstance(
                row[field], bool
            ):
                issues.append(
                    ValidationIssue(
                        display_id, field, "must be a real numeric scalar"
                    )
                )
            elif not _is_finite_real(row[field]):
                issues.append(
                    ValidationIssue(display_id, field, "must be finite")
                )

        for field in row:
            if field not in REQUIRED_FIELDS:
                issues.append(
                    ValidationIssue(
                        display_id, str(field), "is not part of the schema"
                    )
                )

    return ValidationReport(record_count, tuple(issues))