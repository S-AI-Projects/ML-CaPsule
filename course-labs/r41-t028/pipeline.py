"""Naive Bayes text pipeline and error-analysis report for R41-T028.

Task: R41-T028 · 8 of 10
Project/section: Twitter airline sentiment – Naive Bayes
Student: Sharon Robinson
Repository: S-AI-Projects/ML-CaPsule

SYNTHETIC DATA NOTICE
=====================
All default fixture records (t1..t6) and error-analysis reports produced by
this module use six fictional two-word phrases strictly as a pipeline
verification fixture. They are NOT a real Twitter airline sentiment corpus
and must NOT be cited as evidence of real-world or clinical performance.

Design Guarantees
-----------------
1. Explicit train/validation/test separation: Multinomial Naive Bayes
   (Laplace smoothing) fits vocabulary and token counts exclusively on
   ``train`` records (``t1..t4``), keeping ``validation`` (``t5``) and
   ``test`` (``t6``) held out.
2. Structured error-analysis report: every misclassified record emits an
   ``ErrorRow`` with stable ``record_id``, ``expected_label``,
   ``predicted_label``, ``split``, and explicit ``reason_category``; correct
   predictions are excluded from ``error_rows``.
3. Testable error injection: ``inject_deliberate_wrong_prediction`` flips one
   chosen prediction (default ``t5``) so the report pipeline is verified
   end-to-end with nonzero errors and exact aggregate-to-row consistency.
4. Aggregate integrity: ``error_count == len(error_rows)``,
   ``correct_count + error_count == total_evaluated``,
   ``false_positive_count + false_negative_count == error_count``, and
   ``sum(counts_by_reason.values()) == sum(counts_by_split.values()) == error_count``.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

# ── Constants ────────────────────────────────────────────────────────────────

TASK_ID: str = "R41-T028"
PROJECT_SECTION: str = "Twitter airline sentiment – Naive Bayes"
FIXED_SEED: int = 42
VALID_LABELS: frozenset[int] = frozenset({0, 1})
VALID_SPLITS: frozenset[str] = frozenset({"train", "validation", "test"})
REQUIRED_RECORD_FIELDS: tuple[str, ...] = ("record_id", "text", "label")
DEFAULT_INJECTED_RECORD_ID: str = "t5"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

LAB_DIR: Path = Path(__file__).resolve().parent
DEFAULT_FIXTURE_PATH: Path = LAB_DIR / "fixture.json"
DEFAULT_SCHEMA_PATH: Path = LAB_DIR / "schema.json"


# ── Exceptions ───────────────────────────────────────────────────────────────


class ErrorAnalysisPipelineError(ValueError):
    """Base exception for R41-T028 contract violations."""


class SchemaValidationError(ErrorAnalysisPipelineError):
    """Raised when fixture records or predictions violate the schema."""


class SplitContaminationError(ErrorAnalysisPipelineError):
    """Raised when train/validation/test partitions overlap or leak into fit."""


class ModelNotFittedError(RuntimeError):
    """Raised when predict is called before fitting the Naive Bayes model."""


# ── Enums & Immutable Report Structures ──────────────────────────────────────


class ErrorReasonCategory(str, Enum):
    """Controlled vocabulary of error-analysis reason categories."""

    INJECTED_TEST_ERROR = "injected_test_error"
    OOV_TOKEN_SHIFT = "oov_token_shift"
    FALSE_POSITIVE_LEXICAL = "false_positive_lexical"
    FALSE_NEGATIVE_LEXICAL = "false_negative_lexical"


@dataclass(frozen=True)
class SplitPartitions:
    """Validated, pairwise-disjoint train, validation, and test partitions."""

    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    test: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ErrorRow:
    """Single misclassified record entry in the error-analysis report."""

    record_id: str
    split: str
    text: str
    expected_label: int
    predicted_label: int
    reason_category: ErrorReasonCategory
    oov_tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.expected_label == self.predicted_label:
            raise SchemaValidationError(
                f"Correct prediction for {self.record_id!r} must not appear in ErrorRow"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "split": self.split,
            "text": self.text,
            "expected_value": self.expected_label,
            "predicted_value": self.predicted_label,
            "reason_category": self.reason_category.value,
            "oov_tokens": list(self.oov_tokens),
        }


@dataclass(frozen=True)
class ErrorAnalysisReport:
    """Structured error-analysis report with verified aggregate-to-row invariants."""

    total_evaluated: int
    correct_count: int
    error_count: int
    false_positive_count: int
    false_negative_count: int
    counts_by_reason: dict[str, int]
    counts_by_split: dict[str, int]
    error_rows: tuple[ErrorRow, ...]

    def __post_init__(self) -> None:
        if self.error_count != len(self.error_rows):
            raise SchemaValidationError(
                f"Aggregate error_count ({self.error_count}) != detailed rows ({len(self.error_rows)})"
            )
        if self.correct_count + self.error_count != self.total_evaluated:
            raise SchemaValidationError(
                f"correct_count ({self.correct_count}) + error_count ({self.error_count}) "
                f"!= total_evaluated ({self.total_evaluated})"
            )
        if self.false_positive_count + self.false_negative_count != self.error_count:
            raise SchemaValidationError(
                "false_positive_count + false_negative_count != error_count"
            )
        if sum(self.counts_by_reason.values()) != self.error_count:
            raise SchemaValidationError("sum(counts_by_reason) != error_count")
        if sum(self.counts_by_split.values()) != self.error_count:
            raise SchemaValidationError("sum(counts_by_split) != error_count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_evaluated": self.total_evaluated,
            "correct_count": self.correct_count,
            "error_count": self.error_count,
            "false_positive_count": self.false_positive_count,
            "false_negative_count": self.false_negative_count,
            "counts_by_reason": dict(self.counts_by_reason),
            "counts_by_split": dict(self.counts_by_split),
            "error_rows": [row.to_dict() for row in self.error_rows],
        }

    def to_markdown(self) -> str:
        lines = [
            "| record_id | split | text | expected | predicted | reason_category |",
            "|-----------|-------|------|----------|-----------|-----------------|",
        ]
        if not self.error_rows:
            lines.append("| *(none)* | - | - | - | - | *(0 errors)* |")
        else:
            for row in self.error_rows:
                lines.append(
                    f"| `{row.record_id}` | `{row.split}` | `{row.text}` | "
                    f"`{row.expected_label}` | `{row.predicted_label}` | `{row.reason_category.value}` |"
                )
        return "\n".join(lines)


# ── Tokenization & Validation Helpers ────────────────────────────────────────


def tokenize(text: str) -> tuple[str, ...]:
    """Deterministic lowercase alphanumeric tokenizer."""
    if not isinstance(text, str) or not text.strip():
        raise SchemaValidationError(f"text must be a non-empty string, got {text!r}")
    tokens = tuple(TOKEN_PATTERN.findall(text.lower()))
    if not tokens:
        raise SchemaValidationError(f"text {text!r} produced zero tokens")
    return tokens


def _validate_label(value: Any, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value not in VALID_LABELS:
        raise SchemaValidationError(
            f"{context}: label must be an int in {sorted(VALID_LABELS)}, got {value!r}"
        )
    return value


def _validate_record_id(value: Any, row_idx: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(
            f"Row {row_idx}: record_id must be a non-empty string, got {value!r}"
        )
    return value


# ── Bounded Multinomial Naive Bayes Classifier ───────────────────────────────


class MultinomialNaiveBayes:
    """Offline CPU-only Multinomial Naive Bayes classifier (Laplace smoothing)."""

    def __init__(self, *, alpha: float = 1.0, seed: int = FIXED_SEED) -> None:
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(float(alpha)) or float(alpha) <= 0.0:
            raise SchemaValidationError(f"alpha must be positive and finite, got {alpha!r}")
        self.alpha: float = float(alpha)
        self.seed: int = int(seed)
        self.vocabulary_: frozenset[str] = frozenset()
        self.class_priors_: dict[int, float] = {}
        self.token_counts_: dict[int, Counter[str]] = {}
        self.total_tokens_: dict[int, int] = {}
        self.train_record_ids_: frozenset[str] = frozenset()
        self._is_fitted: bool = False

    def fit(self, train_records: Sequence[Mapping[str, Any]]) -> "MultinomialNaiveBayes":
        """Fit vocabulary, class priors, and token counts strictly on training records."""
        if not train_records:
            raise SchemaValidationError("Cannot fit Naive Bayes on empty train_records")

        seen_ids: set[str] = set()
        class_doc_counts: Counter[int] = Counter()
        token_counts: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}
        vocab: set[str] = set()

        for row_idx, rec in enumerate(train_records):
            if not isinstance(rec, Mapping):
                raise SchemaValidationError(f"Row {row_idx}: record must be a mapping")
            if "split" in rec and rec["split"] != "train":
                raise SplitContaminationError(
                    f"Row {row_idx}: cannot fit Naive Bayes on non-train split={rec['split']!r}"
                )
            rec_id = _validate_record_id(rec.get("record_id"), row_idx)
            if rec_id in seen_ids:
                raise SchemaValidationError(f"Duplicate record_id {rec_id!r} in train_records")
            seen_ids.add(rec_id)

            label = _validate_label(rec.get("label"), f"Row {row_idx} ({rec_id!r})")
            tokens = tokenize(rec.get("text"))
            class_doc_counts[label] += 1
            token_counts[label].update(tokens)
            vocab.update(tokens)

        if set(class_doc_counts.keys()) != VALID_LABELS:
            raise SchemaValidationError(
                f"Training records must contain both classes {sorted(VALID_LABELS)}, "
                f"got {sorted(class_doc_counts.keys())}"
            )

        n_docs = float(len(train_records))
        self.vocabulary_ = frozenset(vocab)
        self.class_priors_ = {
            cls: class_doc_counts[cls] / n_docs for cls in sorted(VALID_LABELS)
        }
        self.token_counts_ = token_counts
        self.total_tokens_ = {
            cls: sum(token_counts[cls].values()) for cls in sorted(VALID_LABELS)
        }
        self.train_record_ids_ = frozenset(seen_ids)
        self._is_fitted = True
        return self

    def predict_log_scores(self, text: str) -> dict[int, float]:
        """Return unnormalized log-posterior scores for classes {0, 1}."""
        if not self._is_fitted:
            raise ModelNotFittedError("Call .fit() on training records before predicting")
        tokens = tokenize(text)
        vocab_size = float(len(self.vocabulary_))
        scores: dict[int, float] = {}
        for cls in sorted(VALID_LABELS):
            log_prob = math.log(self.class_priors_[cls])
            denom = float(self.total_tokens_[cls]) + self.alpha * vocab_size
            for tok in tokens:
                if tok in self.vocabulary_:
                    num = float(self.token_counts_[cls][tok]) + self.alpha
                    log_prob += math.log(num / denom)
            scores[cls] = float(log_prob)
        return scores

    def predict_records(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, int]:
        """Predict binary labels mapped by stable record_id."""
        if not self._is_fitted:
            raise ModelNotFittedError("Call .fit() on training records before predicting")
        if not records:
            raise SchemaValidationError("Cannot predict on empty record sequence")

        preds: dict[str, int] = {}
        for row_idx, rec in enumerate(records):
            rec_id = _validate_record_id(rec.get("record_id"), row_idx)
            if rec_id in preds:
                raise SchemaValidationError(f"Duplicate record_id {rec_id!r} in predict batch")
            scores = self.predict_log_scores(rec.get("text"))
            preds[rec_id] = 1 if scores[1] >= scores[0] else 0
        return preds


# ── Error Injection & Error-Analysis Report Builder ──────────────────────────


def inject_deliberate_wrong_prediction(
    predictions_by_id: Mapping[str, int],
    target_record_id: str = DEFAULT_INJECTED_RECORD_ID,
) -> dict[str, int]:
    """Return a copy of predictions with exactly one record's prediction flipped."""
    if target_record_id not in predictions_by_id:
        raise SchemaValidationError(
            f"Cannot inject wrong prediction: record_id {target_record_id!r} not in predictions"
        )
    current = _validate_label(
        predictions_by_id[target_record_id],
        f"prediction[{target_record_id!r}]",
    )
    mutated = dict(predictions_by_id)
    mutated[target_record_id] = 1 - current
    return mutated


def build_error_analysis_report(
    records: Sequence[Mapping[str, Any]],
    predictions_by_id: Mapping[str, int],
    *,
    train_vocab: frozenset[str] | set[str] | None = None,
    injected_error_ids: frozenset[str] | set[str] = frozenset(),
) -> ErrorAnalysisReport:
    """Build a structured error-analysis report with strict aggregate reconciliation."""
    if not records:
        raise SchemaValidationError("Cannot build error-analysis report on empty records")

    record_ids: list[str] = []
    seen_ids: set[str] = set()
    for row_idx, rec in enumerate(records):
        rec_id = _validate_record_id(rec.get("record_id"), row_idx)
        if rec_id in seen_ids:
            raise SchemaValidationError(f"Duplicate record_id {rec_id!r} in records")
        seen_ids.add(rec_id)
        record_ids.append(rec_id)

    missing_preds = sorted(seen_ids - set(predictions_by_id.keys()))
    if missing_preds:
        raise SchemaValidationError(f"Missing predictions for record_id(s): {missing_preds}")
    extra_preds = sorted(set(predictions_by_id.keys()) - seen_ids)
    if extra_preds:
        raise SchemaValidationError(f"Unexpected predictions for unknown record_id(s): {extra_preds}")

    error_rows: list[ErrorRow] = []
    reason_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    fp_count = 0
    fn_count = 0
    correct_count = 0

    for row_idx, rec in enumerate(records):
        rec_id = rec["record_id"]
        split = str(rec.get("split", "unspecified"))
        text = rec.get("text")
        tokens = tokenize(text)
        expected = _validate_label(rec.get("label"), f"Row {row_idx} ({rec_id!r})")
        predicted = _validate_label(
            predictions_by_id[rec_id],
            f"prediction[{rec_id!r}]",
        )

        if predicted == expected:
            correct_count += 1
            continue

        oov = (
            tuple(tok for tok in tokens if tok not in train_vocab)
            if train_vocab is not None
            else ()
        )
        if rec_id in injected_error_ids:
            reason = ErrorReasonCategory.INJECTED_TEST_ERROR
        elif oov:
            reason = ErrorReasonCategory.OOV_TOKEN_SHIFT
        elif predicted == 1 and expected == 0:
            reason = ErrorReasonCategory.FALSE_POSITIVE_LEXICAL
        else:
            reason = ErrorReasonCategory.FALSE_NEGATIVE_LEXICAL

        if predicted == 1 and expected == 0:
            fp_count += 1
        elif predicted == 0 and expected == 1:
            fn_count += 1

        reason_counts[reason.value] += 1
        split_counts[split] += 1
        error_rows.append(
            ErrorRow(
                record_id=rec_id,
                split=split,
                text=text,
                expected_label=expected,
                predicted_label=predicted,
                reason_category=reason,
                oov_tokens=oov,
            )
        )

    return ErrorAnalysisReport(
        total_evaluated=len(records),
        correct_count=correct_count,
        error_count=len(error_rows),
        false_positive_count=fp_count,
        false_negative_count=fn_count,
        counts_by_reason=dict(sorted(reason_counts.items())),
        counts_by_split=dict(sorted(split_counts.items())),
        error_rows=tuple(error_rows),
    )


# ── Fixture Loading & End-to-End Verification ────────────────────────────────


def load_fixture(path: Path | str | None = None) -> dict[str, Any]:
    """Load the six-record fictional text pipeline fixture JSON."""
    fixture_path = Path(path) if path is not None else DEFAULT_FIXTURE_PATH
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not data.get("is_synthetic", False):
        raise SchemaValidationError("Fixture must explicitly declare is_synthetic=true")
    return data


def load_schema(path: Path | str | None = None) -> dict[str, Any]:
    """Load the R41-T028 schema JSON."""
    schema_path = Path(path) if path is not None else DEFAULT_SCHEMA_PATH
    return json.loads(schema_path.read_text(encoding="utf-8"))


def validate_and_partition_fixture(
    records: Sequence[Mapping[str, Any]],
) -> SplitPartitions:
    """Validate schema fields and partition into disjoint train/validation/test splits."""
    if not records:
        raise SchemaValidationError("Fixture records must not be empty")

    buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    record_to_split: dict[str, str] = {}

    for row_idx, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise SchemaValidationError(f"Row {row_idx}: record must be a mapping")
        for field in REQUIRED_RECORD_FIELDS:
            if field not in rec:
                raise SchemaValidationError(f"Row {row_idx}: missing required field {field!r}")
        rec_id = _validate_record_id(rec["record_id"], row_idx)
        tokenize(rec["text"])
        _validate_label(rec["label"], f"Row {row_idx} ({rec_id!r})")

        split = rec.get("split")
        if split not in VALID_SPLITS:
            raise SchemaValidationError(
                f"Row {row_idx} ({rec_id!r}): split must be in {sorted(VALID_SPLITS)}, got {split!r}"
            )
        if rec_id in record_to_split:
            prev = record_to_split[rec_id]
            if prev != split:
                raise SplitContaminationError(
                    f"Contamination detected: record_id {rec_id!r} appears in both {prev!r} and {split!r}"
                )
            raise SchemaValidationError(f"Duplicate record_id {rec_id!r} in split {split!r}")
        record_to_split[rec_id] = split
        buckets[split].append(dict(rec))

    for split_name, rows in buckets.items():
        if not rows:
            raise SchemaValidationError(f"Partition {split_name!r} must not be empty")

    return SplitPartitions(
        train=tuple(buckets["train"]),
        validation=tuple(buckets["validation"]),
        test=tuple(buckets["test"]),
    )


def run_error_analysis_pipeline(
    fixture_path: Path | str | None = None,
    *,
    injected_wrong_id: str = DEFAULT_INJECTED_RECORD_ID,
) -> dict[str, Any]:
    """Fit Naive Bayes on train, evaluate held-out & full splits, and generate reports."""
    fixture_doc = load_fixture(fixture_path)
    partitions = validate_and_partition_fixture(fixture_doc["records"])

    model = MultinomialNaiveBayes(alpha=1.0, seed=FIXED_SEED).fit(partitions.train)
    all_records = partitions.train + partitions.validation + partitions.test
    held_out_records = partitions.validation + partitions.test

    clean_all_preds = model.predict_records(all_records)
    clean_held_out_preds = {r["record_id"]: clean_all_preds[r["record_id"]] for r in held_out_records}

    injected_held_out_preds = inject_deliberate_wrong_prediction(
        clean_held_out_preds,
        target_record_id=injected_wrong_id,
    )
    injected_all_preds = inject_deliberate_wrong_prediction(
        clean_all_preds,
        target_record_id=injected_wrong_id,
    )

    clean_held_out_report = build_error_analysis_report(
        held_out_records,
        clean_held_out_preds,
        train_vocab=model.vocabulary_,
    )
    injected_held_out_report = build_error_analysis_report(
        held_out_records,
        injected_held_out_preds,
        train_vocab=model.vocabulary_,
        injected_error_ids={injected_wrong_id},
    )
    injected_all_report = build_error_analysis_report(
        all_records,
        injected_all_preds,
        train_vocab=model.vocabulary_,
        injected_error_ids={injected_wrong_id},
    )

    return {
        "task_id": TASK_ID,
        "project_section": PROJECT_SECTION,
        "verification_only": True,
        "synthetic_disclaimer": fixture_doc["disclaimer"],
        "seed": FIXED_SEED,
        "train_vocabulary": sorted(model.vocabulary_),
        "splits": {
            "train": [r["record_id"] for r in partitions.train],
            "validation": [r["record_id"] for r in partitions.validation],
            "test": [r["record_id"] for r in partitions.test],
        },
        "clean_predictions": clean_all_preds,
        "injected_wrong_record_id": injected_wrong_id,
        "clean_held_out_report": clean_held_out_report.to_dict(),
        "injected_held_out_report": injected_held_out_report.to_dict(),
        "injected_all_records_report": injected_all_report.to_dict(),
    }
