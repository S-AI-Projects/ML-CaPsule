"""Bounded data contract validator for Netflix titles - EDA (Task R41-T011)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

REQUIRED_FIELDS = ("record_id", "group_id", "x1", "x2", "label")
VALID_LABELS = {0, 1}


@dataclass(frozen=True)
class ValidationFailure:
    """Represents a single schema or data-contract violation without dropping rows."""

    record_id: str
    field: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "record_id": self.record_id,
            "field": self.field,
            "reason": self.reason,
        }


def _is_valid_number(value: Any) -> bool:
    """Accept finite int/float values while rejecting booleans, strings, lists/tensors, and NaN/Inf."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _is_valid_binary_label(value: Any) -> bool:
    """Accept integer 0 or 1 while rejecting booleans, floats, strings, or tensors."""
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return value in VALID_LABELS


def validate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate all classification input records and report every failure without dropping rows."""
    if not isinstance(records, list):
        raise TypeError("records must be provided as a list of dictionaries")

    failures: List[ValidationFailure] = []
    seen_ids: Set[str] = set()

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append(
                ValidationFailure(
                    record_id=f"<row:{index}>",
                    field="record",
                    reason=f"malformed record type: expected dict, got {type(record).__name__}",
                )
            )
            continue

        raw_id = record.get("record_id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            display_id = f"<missing_id_row:{index}>"
            failures.append(
                ValidationFailure(
                    record_id=display_id,
                    field="record_id",
                    reason="missing or malformed text ID: record_id must be a non-empty string",
                )
            )
        else:
            display_id = raw_id.strip()
            if display_id in seen_ids:
                failures.append(
                    ValidationFailure(
                        record_id=display_id,
                        field="record_id",
                        reason=f"duplicate record_id '{display_id}' detected",
                    )
                )
            else:
                seen_ids.add(display_id)

        # Check missing required fields explicitly
        for req_field in REQUIRED_FIELDS:
            if req_field not in record and req_field != "record_id":
                failures.append(
                    ValidationFailure(
                        record_id=display_id,
                        field=req_field,
                        reason=f"missing required field '{req_field}'",
                    )
                )

        if "group_id" in record:
            group_val = record["group_id"]
            if not isinstance(group_val, str) or not group_val.strip():
                failures.append(
                    ValidationFailure(
                        record_id=display_id,
                        field="group_id",
                        reason="malformed text data: group_id must be a non-empty string",
                    )
                )

        for num_field in ("x1", "x2"):
            if num_field in record:
                val = record[num_field]
                if not _is_valid_number(val):
                    failures.append(
                        ValidationFailure(
                            record_id=display_id,
                            field=num_field,
                            reason=(
                                f"malformed numeric/tensor data for '{num_field}': "
                                f"expected finite scalar number, got {type(val).__name__} ({val!r})"
                            ),
                        )
                    )

        if "label" in record:
            lbl = record["label"]
            if not _is_valid_binary_label(lbl):
                failures.append(
                    ValidationFailure(
                        record_id=display_id,
                        field="label",
                        reason=(
                            "malformed label data: expected binary integer 0 or 1, "
                            f"got {type(lbl).__name__} ({lbl!r})"
                        ),
                    )
                )

    return {
        "valid": len(failures) == 0,
        "total_records_checked": len(records),
        "failure_count": len(failures),
        "failures": [f.to_dict() for f in failures],
    }


def validate_stratified_split(
    records: List[Dict[str, Any]], split_map: Dict[str, List[str]]
) -> Dict[str, Any]:
    """Verify explicit disjoint train/validation/test split assignments."""
    train_ids = set(split_map.get("train", []))
    val_ids = set(split_map.get("validation", []))
    test_ids = set(split_map.get("test", []))

    overlaps = (train_ids & val_ids) | (train_ids & test_ids) | (val_ids & test_ids)
    expected_ids = {r["record_id"] for r in records if isinstance(r.get("record_id"), str)}
    assigned_ids = train_ids | val_ids | test_ids

    return {
        "disjoint": len(overlaps) == 0,
        "complete_coverage": assigned_ids == expected_ids,
        "counts": {
            "train": len(train_ids),
            "validation": len(val_ids),
            "test": len(test_ids),
        },
    }


def load_fixture(fixture_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the synthetic pipeline-verification fixture from disk."""
    path = fixture_path or (Path(__file__).resolve().parent / "fixture.json")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


if __name__ == "__main__":
    payload = load_fixture()
    report = validate_records(payload["records"])
    split_report = validate_stratified_split(
        payload["records"], payload["metadata"]["stratified_split"]
    )
    print(json.dumps({"validation": report, "split": split_report}, indent=2))
