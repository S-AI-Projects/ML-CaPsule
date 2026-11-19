"""Data contract validator and baseline k-NN pipeline for MNIST k-NN baseline (Task R41-T021)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REQUIRED_FIELDS = ("record_id", "group_id", "pixels", "label")
VALID_LABELS = {0, 1}


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


def get_environment_info() -> Dict[str, str]:
    """Capture runtime environment versions."""
    return {
        "python": sys.version.split()[0],
        "os": platform.system().lower(),
    }


def _is_valid_scalar_number(value: Any) -> bool:
    """Check if value is a finite int or float (rejecting booleans, strings, lists, NaN, Inf)."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def flatten_pixels(pixels: Any) -> List[float]:
    """Flatten 4x4 matrix or length-16 sequence into a 1D list of floats."""
    if isinstance(pixels, (list, tuple)):
        if len(pixels) == 4 and all(isinstance(r, (list, tuple)) for r in pixels):
            return [float(p) for r in pixels for p in r]
        elif len(pixels) == 16 and all(_is_valid_scalar_number(p) for p in pixels):
            return [float(p) for p in pixels]
    raise ValueError(f"Cannot flatten invalid pixel structure: {type(pixels)}")


def validate_records(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate input records against the image data contract.
    Reports record ID, field, and specific reason for each failure without silently dropping records.
    """
    if not isinstance(records, list):
        raise TypeError("records must be provided as a list of dictionaries")

    failures: List[Dict[str, str]] = []
    seen_ids: Dict[str, int] = {}

    for idx, rec in enumerate(records):
        row_id_fallback = f"row_{idx}"

        if not isinstance(rec, dict):
            failures.append(
                ValidationFailure(
                    record_id=row_id_fallback,
                    field="record",
                    reason=f"Malformed data: record at index {idx} must be a dictionary, got {type(rec).__name__}",
                ).to_dict()
            )
            continue

        current_id = rec.get("record_id")
        id_str = row_id_fallback

        # 1. Validate record_id
        if "record_id" not in rec or current_id is None:
            failures.append(
                ValidationFailure(
                    record_id=row_id_fallback,
                    field="record_id",
                    reason="Missing required field 'record_id'",
                ).to_dict()
            )
        elif not isinstance(current_id, str):
            failures.append(
                ValidationFailure(
                    record_id=str(current_id),
                    field="record_id",
                    reason=f"Malformed text: 'record_id' must be a non-empty string, got {type(current_id).__name__}",
                ).to_dict()
            )
        elif not current_id.strip():
            failures.append(
                ValidationFailure(
                    record_id=row_id_fallback,
                    field="record_id",
                    reason="Malformed text: 'record_id' must not be empty or whitespace",
                ).to_dict()
            )
        else:
            id_str = current_id
            if current_id in seen_ids:
                failures.append(
                    ValidationFailure(
                        record_id=id_str,
                        field="record_id",
                        reason=f"Duplicate record_id '{current_id}' detected (first seen at row {seen_ids[current_id]})",
                    ).to_dict()
                )
            else:
                seen_ids[current_id] = idx

        # 2. Validate group_id
        group_id = rec.get("group_id")
        if "group_id" not in rec or group_id is None:
            failures.append(
                ValidationFailure(
                    record_id=id_str,
                    field="group_id",
                    reason="Missing required field 'group_id'",
                ).to_dict()
            )
        elif not isinstance(group_id, str):
            failures.append(
                ValidationFailure(
                    record_id=id_str,
                    field="group_id",
                    reason=f"Malformed text: 'group_id' must be a non-empty string, got {type(group_id).__name__}",
                ).to_dict()
            )
        elif not group_id.strip():
            failures.append(
                ValidationFailure(
                    record_id=id_str,
                    field="group_id",
                    reason="Malformed text: 'group_id' must not be empty or whitespace",
                ).to_dict()
            )

        # 3. Validate label
        if "label" not in rec or rec.get("label") is None:
            failures.append(
                ValidationFailure(
                    record_id=id_str,
                    field="label",
                    reason="Missing required field 'label'",
                ).to_dict()
            )
        else:
            lbl = rec["label"]
            if isinstance(lbl, bool):
                failures.append(
                    ValidationFailure(
                        record_id=id_str,
                        field="label",
                        reason="Malformed numeric: 'label' must be an integer, not a boolean",
                    ).to_dict()
                )
            elif not isinstance(lbl, int):
                failures.append(
                    ValidationFailure(
                        record_id=id_str,
                        field="label",
                        reason=f"Malformed numeric: 'label' must be an integer, got {type(lbl).__name__}",
                    ).to_dict()
                )
            elif lbl not in VALID_LABELS:
                failures.append(
                    ValidationFailure(
                        record_id=id_str,
                        field="label",
                        reason=f"Malformed numeric: 'label' must be in {{0, 1}}, got {lbl}",
                    ).to_dict()
                )

        # 4. Validate pixels
        if "pixels" not in rec or rec.get("pixels") is None:
            failures.append(
                ValidationFailure(
                    record_id=id_str,
                    field="pixels",
                    reason="Missing required field 'pixels'",
                ).to_dict()
            )
        else:
            px = rec["pixels"]
            if not isinstance(px, (list, tuple)):
                failures.append(
                    ValidationFailure(
                        record_id=id_str,
                        field="pixels",
                        reason=f"Malformed tensor: 'pixels' must be a 4x4 matrix or 16-element sequence, got {type(px).__name__}",
                    ).to_dict()
                )
            else:
                # Check matrix dimensions vs flat sequence
                is_2d = len(px) == 4 and all(isinstance(row, (list, tuple)) for row in px)
                is_1d = len(px) == 16 and all(not isinstance(item, (list, tuple, dict)) for item in px)

                if is_2d:
                    row_len_errors = [len(r) for r in px if len(r) != 4]
                    if row_len_errors:
                        failures.append(
                            ValidationFailure(
                                record_id=id_str,
                                field="pixels",
                                reason="Malformed tensor: 4x4 matrix must have exactly 4 elements per row",
                            ).to_dict()
                        )
                    # Check individual pixel values in 2D
                    for r_idx, row in enumerate(px):
                        if isinstance(row, (list, tuple)):
                            for c_idx, val in enumerate(row):
                                if isinstance(val, bool) or not _is_valid_scalar_number(val):
                                    failures.append(
                                        ValidationFailure(
                                            record_id=id_str,
                                            field="pixels",
                                            reason=f"Malformed numeric: pixel at row {r_idx}, col {c_idx} must be a finite number, got {val}",
                                        ).to_dict()
                                    )
                                elif not (0 <= float(val) <= 255):
                                    failures.append(
                                        ValidationFailure(
                                            record_id=id_str,
                                            field="pixels",
                                            reason=f"Malformed numeric: pixel at row {r_idx}, col {c_idx} must be in range [0, 255], got {val}",
                                        ).to_dict()
                                    )
                elif is_1d:
                    for p_idx, val in enumerate(px):
                        if isinstance(val, bool) or not _is_valid_scalar_number(val):
                            failures.append(
                                ValidationFailure(
                                    record_id=id_str,
                                    field="pixels",
                                    reason=f"Malformed numeric: pixel at index {p_idx} must be a finite number, got {val}",
                                ).to_dict()
                            )
                        elif not (0 <= float(val) <= 255):
                            failures.append(
                                ValidationFailure(
                                    record_id=id_str,
                                    field="pixels",
                                    reason=f"Malformed numeric: pixel at index {p_idx} must be in range [0, 255], got {val}",
                                ).to_dict()
                            )
                else:
                    failures.append(
                        ValidationFailure(
                            record_id=id_str,
                            field="pixels",
                            reason=f"Malformed tensor: 'pixels' dimension error (expected 4x4 nested matrix or 16-element sequence, got shape/length {len(px)})",
                        ).to_dict()
                    )

    return {
        "valid": len(failures) == 0,
        "total_records_checked": len(records),
        "failure_count": len(failures),
        "failures": failures,
    }


class KNNImageClassifier:
    """Bounded, CPU-only k-Nearest Neighbors classifier for 4x4 grayscale images."""

    def __init__(self, k: int = 1):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.train_samples: List[Tuple[List[float], int, str]] = []

    def fit(self, records: List[Dict[str, Any]]) -> "KNNImageClassifier":
        """Store training representations."""
        self.train_samples = []
        for rec in records:
            flat = flatten_pixels(rec["pixels"])
            self.train_samples.append((flat, rec["label"], rec.get("record_id", "")))
        return self

    def predict_one(self, pixels: Any) -> int:
        """Predict binary label for a single image representation."""
        if not self.train_samples:
            raise RuntimeError("Model must be fitted before calling predict")
        flat_query = flatten_pixels(pixels)

        distances = []
        for train_x, train_y, _ in self.train_samples:
            # Euclidean distance
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(flat_query, train_x)))
            distances.append((dist, train_y))

        distances.sort(key=lambda t: t[0])
        k_nearest = distances[: self.k]
        votes = [y for _, y in k_nearest]
        # Return majority label
        return max(set(votes), key=votes.count)

    def predict(self, records: List[Dict[str, Any]]) -> List[int]:
        """Predict labels for a sequence of records."""
        return [self.predict_one(rec["pixels"]) for rec in records]

    def evaluate(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute accuracy and confusion statistics."""
        predictions = self.predict(records)
        true_labels = [rec["label"] for rec in records]
        correct = sum(1 for p, y in zip(predictions, true_labels) if p == y)
        accuracy = correct / len(records) if records else 0.0
        return {
            "total_evaluated": len(records),
            "correct": correct,
            "accuracy": accuracy,
            "predictions": predictions,
            "true_labels": true_labels,
        }


def load_fixture(fixture_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load JSON fixture from disk."""
    if fixture_path is None:
        fixture_path = Path(__file__).parent / "fixture.json"
    with open(fixture_path, "r", encoding="utf-8") as f:
        return json.load(f)
