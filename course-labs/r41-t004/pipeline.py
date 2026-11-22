"""Evaluation metric verification and pipeline module for R41-T004.

Implements and verifies:
- Binary classification metrics: Precision, Recall, F1 Score, Accuracy, Confusion Matrix.
- Explicit definition of averaging conventions (binary positive class).
- Explicit definition of zero-denominator handling (default zero_division=0.0).
- Regression metric: Mean Absolute Error (MAE).
- Strict input validation for types, shapes, and value ranges.
- Data contract and split isolation validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

SPLIT_NAMES = ("train", "validation", "test")
REQUIRED_FIELDS = {"record_id", "group_id", "x1", "x2", "label"}


@dataclass(frozen=True)
class ConfusionMatrix:
    """Confusion matrix counts for binary classification."""

    tp: int
    fp: int
    fn: int
    tn: int
    total: int


@dataclass(frozen=True)
class ClassificationMetrics:
    """Comprehensive classification evaluation summary."""

    precision: float
    recall: float
    f1_score: float
    accuracy: float
    confusion_matrix: ConfusionMatrix


def validate_binary_classification_inputs(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> Tuple[List[int], List[int]]:
    """Validate binary classification inputs.

    Raises
    ------
    ValueError
        If inputs are empty, lengths mismatch, contain non-integer values,
        contain booleans, or contain values other than 0 and 1.
    """
    if y_true is None or y_pred is None:
        raise ValueError("Inputs y_true and y_pred cannot be None.")

    if not isinstance(y_true, (list, tuple)) or not isinstance(y_pred, (list, tuple)):
        raise ValueError("y_true and y_pred must be list or tuple sequences.")

    if len(y_true) == 0:
        raise ValueError("Input sequences cannot be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)}."
        )

    clean_true: List[int] = []
    clean_pred: List[int] = []

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        # In Python, bool subclasses int; strictly reject boolean types
        if isinstance(yt, bool) or not isinstance(yt, int):
            raise ValueError(
                f"y_true[{i}] must be an integer (0 or 1), got {type(yt).__name__} ({yt})."
            )
        if isinstance(yp, bool) or not isinstance(yp, int):
            raise ValueError(
                f"y_pred[{i}] must be an integer (0 or 1), got {type(yp).__name__} ({yp})."
            )

        if yt not in (0, 1):
            raise ValueError(f"y_true[{i}] must be 0 or 1, got {yt}.")
        if yp not in (0, 1):
            raise ValueError(f"y_pred[{i}] must be 0 or 1, got {yp}.")

        clean_true.append(yt)
        clean_pred.append(yp)

    return clean_true, clean_pred


def validate_regression_inputs(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> Tuple[List[float], List[float]]:
    """Validate continuous/numerical regression inputs.

    Raises
    ------
    ValueError
        If inputs are empty, lengths mismatch, contain non-numeric types,
        contain booleans, or contain non-finite numbers (NaN, Inf).
    """
    if y_true is None or y_pred is None:
        raise ValueError("Inputs y_true and y_pred cannot be None.")

    if not isinstance(y_true, (list, tuple)) or not isinstance(y_pred, (list, tuple)):
        raise ValueError("y_true and y_pred must be list or tuple sequences.")

    if len(y_true) == 0:
        raise ValueError("Input sequences cannot be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} elements, y_pred has {len(y_pred)}."
        )

    clean_true: List[float] = []
    clean_pred: List[float] = []

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if isinstance(yt, bool) or not isinstance(yt, (int, float)):
            raise ValueError(
                f"y_true[{i}] must be a numeric scalar, got {type(yt).__name__} ({yt})."
            )
        if isinstance(yp, bool) or not isinstance(yp, (int, float)):
            raise ValueError(
                f"y_pred[{i}] must be a numeric scalar, got {type(yp).__name__} ({yp})."
            )

        f_yt = float(yt)
        f_yp = float(yp)

        if math.isnan(f_yt) or math.isinf(f_yt):
            raise ValueError(f"y_true[{i}] must be a finite real number, got {f_yt}.")
        if math.isnan(f_yp) or math.isinf(f_yp):
            raise ValueError(f"y_pred[{i}] must be a finite real number, got {f_yp}.")

        clean_true.append(f_yt)
        clean_pred.append(f_yp)

    return clean_true, clean_pred


def calculate_confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: int = 1,
) -> ConfusionMatrix:
    """Calculate confusion matrix counts: TP, FP, FN, TN."""
    clean_true, clean_pred = validate_binary_classification_inputs(y_true, y_pred)

    tp = sum(1 for yt, yp in zip(clean_true, clean_pred) if yt == pos_label and yp == pos_label)
    fp = sum(1 for yt, yp in zip(clean_true, clean_pred) if yt != pos_label and yp == pos_label)
    fn = sum(1 for yt, yp in zip(clean_true, clean_pred) if yt == pos_label and yp != pos_label)
    tn = sum(1 for yt, yp in zip(clean_true, clean_pred) if yt != pos_label and yp != pos_label)

    return ConfusionMatrix(tp=tp, fp=fp, fn=fn, tn=tn, total=len(clean_true))


def calculate_precision(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """Calculate binary Precision: TP / (TP + FP).

    Zero-denominator behavior:
    If (TP + FP) == 0 (no positive predictions made), returns zero_division (default 0.0).
    """
    cm = calculate_confusion_matrix(y_true, y_pred, pos_label=pos_label)
    denominator = cm.tp + cm.fp
    if denominator == 0:
        return float(zero_division)
    return float(cm.tp / denominator)


def calculate_recall(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """Calculate binary Recall: TP / (TP + FN).

    Zero-denominator behavior:
    If (TP + FN) == 0 (no positive ground-truth samples), returns zero_division (default 0.0).
    """
    cm = calculate_confusion_matrix(y_true, y_pred, pos_label=pos_label)
    denominator = cm.tp + cm.fn
    if denominator == 0:
        return float(zero_division)
    return float(cm.tp / denominator)


def calculate_f1_score(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> float:
    """Calculate F1 Score: 2 * (Precision * Recall) / (Precision + Recall)."""
    prec = calculate_precision(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    rec = calculate_recall(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)

    denom = prec + rec
    if denom == 0.0:
        return float(zero_division)
    return float(2.0 * (prec * rec) / denom)


def calculate_accuracy(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> float:
    """Calculate Accuracy: (TP + TN) / Total."""
    clean_true, clean_pred = validate_binary_classification_inputs(y_true, y_pred)
    correct = sum(1 for yt, yp in zip(clean_true, clean_pred) if yt == yp)
    return float(correct / len(clean_true))


def calculate_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    pos_label: int = 1,
    zero_division: float = 0.0,
) -> ClassificationMetrics:
    """Calculate all classification metrics with explicit averaging and zero-division handling."""
    cm = calculate_confusion_matrix(y_true, y_pred, pos_label=pos_label)
    prec = calculate_precision(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    rec = calculate_recall(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    f1 = calculate_f1_score(y_true, y_pred, pos_label=pos_label, zero_division=zero_division)
    acc = calculate_accuracy(y_true, y_pred)

    return ClassificationMetrics(
        precision=prec,
        recall=rec,
        f1_score=f1,
        accuracy=acc,
        confusion_matrix=cm,
    )


def calculate_mae(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> float:
    """Calculate regression Mean Absolute Error (MAE): (1/N) * sum(|y_true - y_pred|)."""
    clean_true, clean_pred = validate_regression_inputs(y_true, y_pred)
    total_abs_error = sum(abs(yt - yp) for yt, yp in zip(clean_true, clean_pred))
    return float(total_abs_error / len(clean_true))


def validate_records(records: Sequence[Any]) -> None:
    """Validate records against schema (record_id, group_id, x1, x2, label)."""
    if not records:
        raise ValueError("Records must not be empty.")

    if not isinstance(records, (list, tuple)):
        raise ValueError("Records must be a list or tuple of dictionaries.")

    seen_ids: Set[str] = set()

    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {i} must be a dictionary.")

        missing = REQUIRED_FIELDS - set(record.keys())
        if missing:
            raise ValueError(f"Record at index {i} is missing fields: {sorted(missing)}.")

        rid = record["record_id"]
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError(f"record_id at index {i} must be a non-empty string.")
        if rid in seen_ids:
            raise ValueError(f"Duplicate record_id '{rid}' detected at index {i}.")
        seen_ids.add(rid)

        gid = record["group_id"]
        if not isinstance(gid, str) or not gid.strip():
            raise ValueError(f"group_id at index {i} must be a non-empty string.")

        for fld in ("x1", "x2"):
            val = record[fld]
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{fld} at index {i} must be a numeric scalar.")

        lbl = record["label"]
        if isinstance(lbl, bool) or not isinstance(lbl, int) or lbl not in (0, 1):
            raise ValueError(f"label at index {i} must be binary int (0 or 1).")


def validate_split_manifest(
    records: Sequence[Dict[str, Any]],
    split_manifest: Dict[str, Sequence[str]],
) -> None:
    """Validate split manifest to ensure train/validation/test separation without leakage."""
    if not isinstance(split_manifest, dict):
        raise ValueError("split_manifest must be a dictionary.")

    if set(SPLIT_NAMES) != set(split_manifest.keys()):
        raise ValueError(f"Manifest must contain exactly {SPLIT_NAMES}.")

    all_ids = {r["record_id"] for r in records}
    seen_ids: Set[str] = set()

    for name in SPLIT_NAMES:
        split_ids = split_manifest[name]
        if not isinstance(split_ids, (list, tuple)):
            raise ValueError(f"Split '{name}' IDs must be a list or tuple.")

        if len(split_ids) != len(set(split_ids)):
            raise ValueError(f"Duplicate record_id within split '{name}'.")

        unknown = set(split_ids) - all_ids
        if unknown:
            raise ValueError(f"Split '{name}' contains unknown record_ids: {sorted(unknown)}.")

        overlap = seen_ids.intersection(split_ids)
        if overlap:
            raise ValueError(f"Leakage: record_ids {sorted(overlap)} overlap across splits.")

        seen_ids.update(split_ids)

    if seen_ids != all_ids:
        missing = all_ids - seen_ids
        raise ValueError(f"Every record must be assigned. Missing: {sorted(missing)}.")
