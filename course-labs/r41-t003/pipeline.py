"""Majority-class baseline classifier and validation pipeline for R41-T003.

This module provides the simplest honest baseline for text classification:
- Determines the majority class using training labels ONLY.
- Applies deterministic tie-breaking when training classes have equal frequencies.
- Evaluates held-out accuracy and error metrics on validation and test sets separately.
- Strictly validates the schema contract and split integrity (zero leakage).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set

SPLIT_NAMES = ("train", "validation", "test")
REQUIRED_FIELDS = {"record_id", "text", "label"}


@dataclass
class EvaluationMetric:
    """Evaluation metrics computed on held-out records."""

    total: int
    correct: int
    incorrect: int
    accuracy: float
    error_rate: float
    predictions: List[int]
    true_labels: List[int]


@dataclass
class BaselinePipelineResult:
    """Full outcome of fitting and evaluating the majority-class baseline."""

    majority_class: int
    train_class_counts: Dict[int, int]
    tie_detected: bool
    train_record_ids: List[str]
    validation_metrics: EvaluationMetric
    test_metrics: EvaluationMetric
    overall_held_out_metrics: EvaluationMetric


def validate_records(records: Sequence[Any]) -> None:
    """Validate records against the (record_id, text, label) data contract.

    Raises
    ------
    ValueError
        If records are empty, malformed, missing required fields, contain invalid types,
        or have duplicate record_ids.
    """
    if not records:
        raise ValueError("Records collection must not be empty.")

    if not isinstance(records, (list, tuple)):
        raise ValueError("Records must be a list or tuple of dictionaries.")

    seen_ids: Set[str] = set()

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Record at index {idx} must be a dictionary, got {type(record).__name__}.")

        missing = REQUIRED_FIELDS - set(record.keys())
        if missing:
            raise ValueError(f"Record at index {idx} is missing required fields: {sorted(missing)}.")

        record_id = record["record_id"]
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"record_id at index {idx} must be a non-empty string.")

        if record_id in seen_ids:
            raise ValueError(f"Duplicate record_id '{record_id}' detected at index {idx}.")
        seen_ids.add(record_id)

        text = record["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"text field at index {idx} must be a non-empty string.")

        label = record["label"]
        if isinstance(label, bool) or not isinstance(label, int):
            raise ValueError(f"label at index {idx} must be an integer (0 or 1), got {type(label).__name__}.")

        if label not in (0, 1):
            raise ValueError(f"label at index {idx} must be 0 or 1, got {label}.")


def validate_split_manifest(
    records: Sequence[Dict[str, Any]],
    split_manifest: Dict[str, Sequence[str]],
) -> None:
    """Validate train/validation/test split manifest to prevent data leakage and overlap.

    Raises
    ------
    ValueError
        If the manifest is missing splits, contains unknown record IDs, has overlaps
        between splits, has an empty training split, or does not assign every record.
    """
    if not isinstance(split_manifest, dict):
        raise ValueError("split_manifest must be a dictionary.")

    manifest_splits = set(split_manifest.keys())
    if set(SPLIT_NAMES) != manifest_splits:
        raise ValueError(f"split_manifest must contain exactly {SPLIT_NAMES}, got {sorted(manifest_splits)}.")

    all_record_ids = {r["record_id"] for r in records}
    seen_in_splits: Set[str] = set()

    for split_name in SPLIT_NAMES:
        split_ids = split_manifest[split_name]
        if not isinstance(split_ids, (list, tuple)):
            raise ValueError(f"Split '{split_name}' must be a list or tuple of record_ids.")

        # Check for duplicates within the same split
        if len(split_ids) != len(set(split_ids)):
            raise ValueError(f"Duplicate record_id found within split '{split_name}'.")

        # Check for unknown record IDs
        unknown = set(split_ids) - all_record_ids
        if unknown:
            raise ValueError(f"Split '{split_name}' contains unknown record_ids: {sorted(unknown)}.")

        # Check for overlap with previously seen splits (data leakage)
        overlap = seen_in_splits.intersection(split_ids)
        if overlap:
            raise ValueError(f"Data leakage detected: record_ids {sorted(overlap)} overlap across splits.")

        seen_in_splits.update(split_ids)

    # Ensure training split is not empty
    if not split_manifest["train"]:
        raise ValueError("Training split cannot be empty.")

    # Ensure all records are assigned exactly once
    if seen_in_splits != all_record_ids:
        missing_ids = all_record_ids - seen_in_splits
        raise ValueError(f"Manifest must partition every record. Unassigned: {sorted(missing_ids)}.")


def calculate_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> EvaluationMetric:
    """Calculate accuracy and error metrics on held-out predictions."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) lengths must match.")

    total = len(y_true)
    if total == 0:
        return EvaluationMetric(
            total=0,
            correct=0,
            incorrect=0,
            accuracy=0.0,
            error_rate=0.0,
            predictions=[],
            true_labels=[],
        )

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    incorrect = total - correct
    accuracy = correct / total
    error_rate = incorrect / total

    return EvaluationMetric(
        total=total,
        correct=correct,
        incorrect=incorrect,
        accuracy=accuracy,
        error_rate=error_rate,
        predictions=list(y_pred),
        true_labels=list(y_true),
    )


class MajorityClassBaseline:
    """Majority-class baseline classifier for binary classification text contracts."""

    def __init__(self, tie_breaker: str = "lowest") -> None:
        """Initialize the baseline.

        Parameters
        ----------
        tie_breaker : str
            Strategy when training classes have equal counts.
            'lowest' (default) chooses min(tied_classes), e.g. 0.
            'highest' chooses max(tied_classes), e.g. 1.
        """
        if tie_breaker not in ("lowest", "highest"):
            raise ValueError(f"Unknown tie_breaker '{tie_breaker}'. Choose 'lowest' or 'highest'.")

        self.tie_breaker = tie_breaker
        self.majority_class: Optional[int] = None
        self.class_counts_: Dict[int, int] = {}
        self.tie_detected_: bool = False
        self.is_fitted_: bool = False

    def fit(self, train_records: Sequence[Dict[str, Any]]) -> MajorityClassBaseline:
        """Fit baseline using training labels ONLY.

        Never inspects or uses held-out (validation or test) records.
        """
        if not train_records:
            raise ValueError("Cannot fit baseline on empty training records.")

        labels = [r["label"] for r in train_records]
        counts = Counter(labels)
        self.class_counts_ = dict(counts)

        max_count = max(counts.values())
        top_classes = [cls for cls, count in counts.items() if count == max_count]

        if len(top_classes) > 1:
            self.tie_detected_ = True
            if self.tie_breaker == "lowest":
                self.majority_class = min(top_classes)
            else:
                self.majority_class = max(top_classes)
        else:
            self.tie_detected_ = False
            self.majority_class = top_classes[0]

        self.is_fitted_ = True
        return self

    def predict(self, records: Sequence[Dict[str, Any]]) -> List[int]:
        """Predict the determined majority class for every candidate record."""
        if not self.is_fitted_ or self.majority_class is None:
            raise RuntimeError("Baseline must be fitted on training records before predicting.")

        return [self.majority_class] * len(records)

    def evaluate(self, held_out_records: Sequence[Dict[str, Any]]) -> EvaluationMetric:
        """Evaluate accuracy and error metrics on held-out records."""
        if not self.is_fitted_ or self.majority_class is None:
            raise RuntimeError("Baseline must be fitted before evaluation.")

        y_true = [r["label"] for r in held_out_records]
        y_pred = self.predict(held_out_records)
        return calculate_metrics(y_true, y_pred)


def run_baseline_pipeline(
    records: Sequence[Dict[str, Any]],
    split_manifest: Dict[str, Sequence[str]],
    tie_breaker: str = "lowest",
) -> BaselinePipelineResult:
    """Execute end-to-end baseline training and held-out evaluation."""
    validate_records(records)
    validate_split_manifest(records, split_manifest)

    record_map = {r["record_id"]: r for r in records}

    train_recs = [record_map[rid] for rid in split_manifest["train"]]
    val_recs = [record_map[rid] for rid in split_manifest["validation"]]
    test_recs = [record_map[rid] for rid in split_manifest["test"]]
    combined_held_out = val_recs + test_recs

    # Fit ONLY on training records
    baseline = MajorityClassBaseline(tie_breaker=tie_breaker).fit(train_recs)

    # Evaluate held-out splits
    val_metrics = baseline.evaluate(val_recs)
    test_metrics = baseline.evaluate(test_recs)
    overall_metrics = baseline.evaluate(combined_held_out)

    return BaselinePipelineResult(
        majority_class=baseline.majority_class,  # type: ignore[arg-type]
        train_class_counts=baseline.class_counts_,
        tie_detected=baseline.tie_detected_,
        train_record_ids=list(split_manifest["train"]),
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        overall_held_out_metrics=overall_metrics,
    )
