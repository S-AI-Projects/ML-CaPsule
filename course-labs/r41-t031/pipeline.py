"""Data contract validator and PCA projection pipeline for sklearn digits – PCA + SVM (Task R41-T031)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REQUIRED_FIELDS = ("record_id", "x1", "x2")


@dataclass(frozen=True)
class ValidationFailure:
    """Represents a single schema validation failure without dropping records."""

    record_id: str
    field: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "record_id": self.record_id,
            "field": self.field,
            "reason": self.reason,
        }


def _is_valid_scalar_number(value: Any) -> bool:
    """Check if value is a finite int or float (rejecting booleans, strings, lists, tensors, NaN, Inf)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def validate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate input records against the projection data contract.
    Reports record ID, field, and specific reason for each failure without silently dropping records.
    """
    failures: List[Dict[str, str]] = []
    seen_ids: Dict[str, int] = {}

    for idx, rec in enumerate(records):
        current_id = rec.get("record_id")
        id_str = f"row_{idx}"

        # 1. Check record_id
        if "record_id" not in rec or current_id is None:
            failures.append(ValidationFailure(id_str, "record_id", "Missing required field 'record_id'").to_dict())
        elif not isinstance(current_id, str) or not current_id.strip():
            failures.append(ValidationFailure(str(current_id), "record_id", "Malformed text: record_id must be a non-empty string").to_dict())
        else:
            id_str = current_id
            if current_id in seen_ids:
                failures.append(ValidationFailure(id_str, "record_id", f"Duplicate record_id '{current_id}' detected").to_dict())
            else:
                seen_ids[current_id] = idx

        # 2. Check feature fields x1, x2
        for field in ("x1", "x2"):
            if field not in rec or rec[field] is None:
                failures.append(ValidationFailure(id_str, field, f"Missing required numeric feature '{field}'").to_dict())
            elif isinstance(rec[field], (list, tuple, dict)):
                failures.append(ValidationFailure(id_str, field, f"Malformed tensor/collection: '{field}' must be a scalar float/int, not a sequence/tensor").to_dict())
            elif isinstance(rec[field], str):
                failures.append(ValidationFailure(id_str, field, f"Malformed text: '{field}' must be numeric, got string '{rec[field]}'").to_dict())
            elif not _is_valid_scalar_number(rec[field]):
                failures.append(ValidationFailure(id_str, field, f"Malformed numeric: '{field}' must be a finite float/int").to_dict())

    return {
        "valid": len(failures) == 0,
        "total_records_checked": len(records),
        "failure_count": len(failures),
        "failures": failures,
    }


def project_and_reconstruct(
    points: List[Tuple[float, float]],
    unit_vector: Optional[Tuple[float, float]] = None,
    tolerance: float = 1e-6,
) -> List[Dict[str, Any]]:
    """
    Projects 2D centered collinear points onto a unit vector and reconstructs them.
    Default unit vector is (1/sqrt(2), 1/sqrt(2)).
    Recovers each point within tolerance (allowing for sign reversal of learned component).
    """
    if unit_vector is None:
        u = 1.0 / math.sqrt(2.0)
        unit_vector = (u, u)

    u1, u2 = unit_vector
    norm = math.sqrt(u1 * u1 + u2 * u2)
    u1, u2 = u1 / norm, u2 / norm

    results = []
    for x1, x2 in points:
        # Scalar projection
        scalar_proj = x1 * u1 + x2 * u2
        # Reconstruction
        recon_x1 = scalar_proj * u1
        recon_x2 = scalar_proj * u2

        # Error check (accounting for potential sign flip if learned component is reversed)
        err = math.sqrt((x1 - recon_x1) ** 2 + (x2 - recon_x2) ** 2)
        recovered = err <= tolerance

        results.append({
            "original": (x1, x2),
            "scalar_projection": scalar_proj,
            "reconstruction": (recon_x1, recon_x2),
            "reconstruction_error": err,
            "recovered": recovered,
        })

    return results


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load the JSON fixture file."""
    if path is None:
        path = Path(__file__).parent / "fixture.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
