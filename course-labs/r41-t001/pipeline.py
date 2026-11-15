"""Data contract schema validator for classification input fields (R41-T001).

Validates records against the strict classification schema:
- record_id: non-empty string, unique across dataset
- group_id: non-empty string
- x1: scalar numeric (int or float, not bool, not sequence/tensor, finite)
- x2: scalar numeric (int or float, not bool, not sequence/tensor, finite)
- label: binary integer (0 or 1, not bool, not float)

Reports record ID, field name, and specific failure reason without silently
dropping records.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set


@dataclass(frozen=True)
class ValidationFailure:
    """Represents a single schema validation failure for a field in a record."""

    record_id: str
    field: str
    reason: str


@dataclass
class DatasetValidationResult:
    """Summary of schema validation across an entire dataset of records."""

    all_valid: bool
    failures: List[ValidationFailure] = field(default_factory=list)
    valid_count: int = 0
    invalid_count: int = 0
    total_records: int = 0


def _is_sequence_or_tensor(val: Any) -> bool:
    """Return True if val is a sequence, collection, or tensor-like container."""
    if isinstance(val, (list, tuple, dict, set, frozenset, bytearray)):
        return True
    # Catch any custom or third-party tensor-like types (e.g., torch.Tensor, np.ndarray)
    # by inspecting class name or shape attribute without requiring external imports
    if hasattr(val, "shape") or hasattr(val, "ndim"):
        return True
    return False


def validate_single_record(
    record: Any,
    row_index: int = 0,
    seen_record_ids: Optional[Dict[str, int]] = None,
) -> List[ValidationFailure]:
    """Validate a single record against schema rules and return all failures found.

    Multiple failures in one record are all captured and returned.
    """
    failures: List[ValidationFailure] = []

    if not isinstance(record, dict):
        return [
            ValidationFailure(
                record_id=f"<row_{row_index}>",
                field="<record>",
                reason=f"Record must be a dictionary, got {type(record).__name__}",
            )
        ]

    # -------------------------------------------------------------------------
    # 1. Validate 'record_id'
    # -------------------------------------------------------------------------
    rec_id_val = record.get("record_id")
    has_valid_rec_id = False
    effective_rec_id: str

    if "record_id" not in record:
        effective_rec_id = "<missing>"
        failures.append(
            ValidationFailure(
                record_id=effective_rec_id,
                field="record_id",
                reason="Missing required field 'record_id'",
            )
        )
    elif rec_id_val is None:
        effective_rec_id = "<missing>"
        failures.append(
            ValidationFailure(
                record_id=effective_rec_id,
                field="record_id",
                reason="'record_id' cannot be None",
            )
        )
    elif not isinstance(rec_id_val, str):
        effective_rec_id = str(rec_id_val)
        failures.append(
            ValidationFailure(
                record_id=effective_rec_id,
                field="record_id",
                reason=f"'record_id' must be a string, got {type(rec_id_val).__name__}",
            )
        )
    elif not rec_id_val.strip():
        effective_rec_id = "<empty>"
        failures.append(
            ValidationFailure(
                record_id=effective_rec_id,
                field="record_id",
                reason="'record_id' cannot be empty or whitespace-only",
            )
        )
    else:
        effective_rec_id = rec_id_val.strip()
        has_valid_rec_id = True

        # Check for duplicates across the dataset
        if seen_record_ids is not None:
            if effective_rec_id in seen_record_ids:
                prior_idx = seen_record_ids[effective_rec_id]
                failures.append(
                    ValidationFailure(
                        record_id=effective_rec_id,
                        field="record_id",
                        reason=(
                            f"Duplicate record_id '{effective_rec_id}' (previously seen at record index {prior_idx})"
                        ),
                    )
                )
            else:
                seen_record_ids[effective_rec_id] = row_index

    # -------------------------------------------------------------------------
    # 2. Validate 'group_id'
    # -------------------------------------------------------------------------
    if "group_id" not in record:
        failures.append(
            ValidationFailure(
                record_id=effective_rec_id,
                field="group_id",
                reason="Missing required field 'group_id'",
            )
        )
    else:
        grp_val = record.get("group_id")
        if grp_val is None:
            failures.append(
                ValidationFailure(
                    record_id=effective_rec_id,
                    field="group_id",
                    reason="'group_id' cannot be None",
                )
            )
        elif not isinstance(grp_val, str):
            failures.append(
                ValidationFailure(
                    record_id=effective_rec_id,
                    field="group_id",
                    reason=f"'group_id' must be a string, got {type(grp_val).__name__}",
                )
            )
        elif not grp_val.strip():
            failures.append(
                ValidationFailure(
                    record_id=effective_rec_id,
                    field="group_id",
                    reason="'group_id' cannot be empty or whitespace-only",
                )
            )

    # -------------------------------------------------------------------------
    # 3. Validate numeric scalar fields: 'x1' and 'x2'
    # -------------------------------------------------------------------------
    for field_name in ("x1", "x2"):
        if field_name not in record:
            failures.append(
                ValidationFailure(
                    record_id=effective_rec_id,
                    field=field_name,
                    reason=f"Missing required field '{field_name}'",
                )
            )
            continue

        val = record.get(field_name)

        if val is None:
            failures.append(
                ValidationFailure(
                    record_id=effective_rec_id,
                    field=field_name,
                    reason=f"'{field_name}' cannot be None",
                )
            )
        elif isinstance(val, bool):
            # In Python, bool subclasses int, so isinstance(True, int) is True!
            # We must explicitly reject bool as a numeric scalar.
            failures.append(
                ValidationFailure(
                    record_id=effective_rec_id,
                    field=field_name,
                    reason=f"'{field_name}' must be a numeric scalar (int or float), got boolean {val}",
                )
            )
        elif _is_sequence_or_tensor(val):
            failures.append(
                ValidationFailure(
                    record_id=effective_rec_id,
                    field=field_name,
                    reason=f"'{field_name}' must be a scalar number, got sequence/tensor of type {type(val).__name__}",
                )
            )
        elif not isinstance(val, (int, float)):
            failures.append(
                ValidationFailure(
                    record_id=effective_rec_id,
                    field=field_name,
                    reason=f"'{field_name}' must be a numeric scalar, got {type(val).__name__}",
                )
            )
        elif isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            failures.append(
                ValidationFailure(
                    record_id=effective_rec_id,
                    field=field_name,
                    reason=f"'{field_name}' must be a finite real number, got {val}",
                )
            )

    # -------------------------------------------------------------------------
    # 4. Validate 'label' (binary integer: 0 or 1)
    # -------------------------------------------------------------------------
    if "label" not in record:
        failures.append(
            ValidationFailure(
                record_id=effective_rec_id,
                field="label",
                reason="Missing required field 'label'",
            )
        )
    else:
        label_val = record.get("label")
        if label_val is None:
            failures.append(
                ValidationFailure(
                    record_id=effective_rec_id,
                    field="label",
                    reason="'label' cannot be None",
                )
            )
        elif isinstance(label_val, bool):
            failures.append(
                ValidationFailure(
                    record_id=effective_rec_id,
                    field="label",
                    reason=f"'label' must be a binary integer (0 or 1), got boolean {label_val}",
                )
            )
        elif not isinstance(label_val, int):
            failures.append(
                ValidationFailure(
                    record_id=effective_rec_id,
                    field="label",
                    reason=f"'label' must be an integer, got {type(label_val).__name__}",
                )
            )
        elif label_val not in (0, 1):
            failures.append(
                ValidationFailure(
                    record_id=effective_rec_id,
                    field="label",
                    reason=f"'label' must be binary (0 or 1), got {label_val}",
                )
            )

    return failures


def validate_dataset(records: Sequence[Any]) -> DatasetValidationResult:
    """Validate a sequence of records against the classification schema contract.

    Parameters
    ----------
    records : Sequence[Any]
        A list/sequence of record dictionaries to validate.

    Returns
    -------
    DatasetValidationResult
        Detailed report containing overall validity, all failures, and record counts.
        Records are never silently dropped or filtered out.
    """
    all_failures: List[ValidationFailure] = []
    seen_ids: Dict[str, int] = {}
    valid_count = 0
    invalid_count = 0

    for idx, rec in enumerate(records):
        rec_failures = validate_single_record(rec, row_index=idx, seen_record_ids=seen_ids)
        if rec_failures:
            invalid_count += 1
            all_failures.extend(rec_failures)
        else:
            valid_count += 1

    return DatasetValidationResult(
        all_valid=(len(all_failures) == 0),
        failures=all_failures,
        valid_count=valid_count,
        invalid_count=invalid_count,
        total_records=len(records),
    )

