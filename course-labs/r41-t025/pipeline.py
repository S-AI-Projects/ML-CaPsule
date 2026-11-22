"""Numeric preprocessing contract for R41-T025 (Zoo animals – decision tree).

Task: R41-T025 · 5 of 10
Student: Sharon Robinson
Repository: S-AI-Projects/ML-CaPsule

SYNTHETIC DATA NOTICE
=====================
All default fixture records and verification outputs produced by this module
use an eight-row synthetic pipeline-verification fixture. They are NOT
observations from the UCI Zoo dataset or any other real-world dataset, and
must NOT be cited as evidence of real-world or clinical performance.

Design Guarantees
-----------------
1. Training-only fitting: ``fit_numeric_preprocessor`` computes numeric
   standardization statistics (mean, population std, safe scale) exclusively
   from training rows (``split == "train"`` when present).
2. Transform-only public interface: fitting returns a frozen, immutable
   ``FittedNumericPreprocessor`` that exposes only ``transform`` and
   ``transform_split`` — no ``fit``, ``partial_fit``, or ``fit_transform``
   method exists on the fitted transformer, making fitting on held-out input
   impossible through the public transform interface.
3. Explicit validation contracts:
   - Feature order validation (exact ordered match against ``("x1", "x2")``;
     rejects swapped, missing, or unexpected feature columns).
   - Label-set validation (strict integer membership in ``{0, 1}``; rejects
     booleans, floats, strings, ``None``, and out-of-set integers).
   - Missing-value policy validation (``MissingValuePolicy.REJECT`` by default,
     or explicit ``MissingValuePolicy.IMPUTE_TRAIN_MEAN`` using training-only
     means; rejects unknown policies and non-finite values).
4. Contamination rejection:
   - Rejects non-train rows passed to ``fit_numeric_preprocessor``.
   - Rejects overlapping ``record_id`` or ``group_id`` across ``train``,
     ``validation``, and ``test`` partitions.
   - Rejects training ``record_id`` or ``group_id`` appearing in held-out
     ``validation`` or ``test`` transform calls.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

# ── Constants ────────────────────────────────────────────────────────────────

TASK_ID: str = "R41-T025"
PROJECT_SECTION: str = "Zoo animals – decision tree"
FIXED_SEED: int = 42
EXPECTED_FEATURE_ORDER: tuple[str, ...] = ("x1", "x2")
VALID_LABELS: frozenset[int] = frozenset({0, 1})
VALID_SPLITS: frozenset[str] = frozenset({"train", "validation", "test"})
REQUIRED_RECORD_FIELDS: tuple[str, ...] = (
    "record_id",
    "group_id",
    "x1",
    "x2",
    "label",
)
METADATA_FIELDS: frozenset[str] = frozenset(
    {"record_id", "group_id", "label", "split"}
)
OUTPUT_DTYPE: str = "float64"

LAB_DIR: Path = Path(__file__).resolve().parent
DEFAULT_FIXTURE_PATH: Path = LAB_DIR / "fixture.json"
DEFAULT_SCHEMA_PATH: Path = LAB_DIR / "schema.json"


# ── Exceptions ───────────────────────────────────────────────────────────────


class PreprocessingError(ValueError):
    """Base exception for preprocessing contract violations."""


class SchemaValidationError(PreprocessingError):
    """Raised when record identifiers or required fields violate the schema."""


class FeatureOrderError(PreprocessingError):
    """Raised when feature names or column order do not match the contract."""


class InvalidLabelError(PreprocessingError):
    """Raised when labels contain values outside the allowed integer set."""


class MissingValueError(PreprocessingError):
    """Raised when missing or non-finite values violate the active policy."""


class InvalidPolicyError(PreprocessingError):
    """Raised when an unsupported missing-value policy is requested."""


class ContaminationError(PreprocessingError):
    """Raised when train/validation/test separation is contaminated."""


# ── Policies & Immutable Types ───────────────────────────────────────────────


class MissingValuePolicy(str, Enum):
    """Explicit missing-value handling policies."""

    REJECT = "reject"
    IMPUTE_TRAIN_MEAN = "impute_train_mean"


@dataclass(frozen=True)
class TransformedMatrix:
    """Immutable numeric output matrix with explicit shape and float64 dtype."""

    values: tuple[tuple[float, ...], ...]
    shape: tuple[int, int]
    dtype: str
    feature_names: tuple[str, ...]
    record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.dtype != OUTPUT_DTYPE:
            raise SchemaValidationError(
                f"Expected dtype {OUTPUT_DTYPE!r}, got {self.dtype!r}"
            )
        n_rows = len(self.values)
        n_cols = len(self.feature_names)
        if self.shape != (n_rows, n_cols):
            raise SchemaValidationError(
                f"Shape mismatch: declared {self.shape}, actual {(n_rows, n_cols)}"
            )
        if len(self.record_ids) != n_rows:
            raise SchemaValidationError(
                f"record_ids length {len(self.record_ids)} != row count {n_rows}"
            )
        for r_idx, row in enumerate(self.values):
            if len(row) != n_cols:
                raise SchemaValidationError(
                    f"Row {r_idx} has {len(row)} columns, expected {n_cols}"
                )
            for c_idx, val in enumerate(row):
                if isinstance(val, bool) or not isinstance(val, float):
                    raise SchemaValidationError(
                        f"Value at ({r_idx}, {c_idx}) must be float64, got {type(val).__name__}"
                    )
                if not math.isfinite(val):
                    raise MissingValueError(
                        f"Non-finite output value at ({r_idx}, {c_idx}): {val!r}"
                    )

    def to_lists(self) -> list[list[float]]:
        """Return a mutable list-of-lists copy of the float64 values."""
        return [list(row) for row in self.values]

    def to_numpy(self) -> Any:
        """Convert to a numpy float64 ndarray when NumPy is installed."""
        import numpy as np  # optional convenience only; stdlib path never needs it

        return np.asarray(self.values, dtype=np.float64)


@dataclass(frozen=True)
class SplitPartitions:
    """Validated, pairwise-disjoint train, validation, and test partitions."""

    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    test: tuple[dict[str, Any], ...]


# ── Validation Helpers ───────────────────────────────────────────────────────


def _resolve_policy(
    policy: MissingValuePolicy | str,
) -> MissingValuePolicy:
    """Validate and normalize the missing-value policy."""
    if isinstance(policy, MissingValuePolicy):
        return policy
    if isinstance(policy, str):
        try:
            return MissingValuePolicy(policy)
        except ValueError as exc:
            raise InvalidPolicyError(
                f"Unsupported missing-value policy {policy!r}. "
                f"Allowed: {[p.value for p in MissingValuePolicy]}"
            ) from exc
    raise InvalidPolicyError(
        f"Missing-value policy must be MissingValuePolicy or str, got {type(policy).__name__}"
    )


def _validate_non_empty_id(value: Any, field_name: str, row_idx: int) -> str:
    """Validate stable string identifiers such as record_id and group_id."""
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(
            f"Row {row_idx}: {field_name!r} must be a non-empty string, got {value!r}"
        )
    return value


def _validate_label(value: Any, valid_labels: frozenset[int], row_idx: int) -> int:
    """Validate strict integer label membership (rejects bool/float/None/str)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidLabelError(
            f"Row {row_idx}: label must be an int in {sorted(valid_labels)}, "
            f"got {value!r} ({type(value).__name__})"
        )
    if value not in valid_labels:
        raise InvalidLabelError(
            f"Row {row_idx}: label {value!r} is outside allowed label set {sorted(valid_labels)}"
        )
    return value


def _is_missing_value(value: Any) -> bool:
    """Return True if a value represents a missing/NaN entry."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "na", "null", "none"}:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _parse_numeric_feature(
    value: Any,
    feature_name: str,
    row_idx: int,
    policy: MissingValuePolicy,
) -> float | None:
    """Parse a feature cell into finite float64 or None (when imputation is active)."""
    if _is_missing_value(value):
        if policy is MissingValuePolicy.REJECT:
            raise MissingValueError(
                f"Row {row_idx}, feature {feature_name!r}: missing value {value!r} "
                f"rejected under policy={policy.value!r}"
            )
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MissingValueError(
            f"Row {row_idx}, feature {feature_name!r}: expected finite numeric value, "
            f"got {value!r} ({type(value).__name__})"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise MissingValueError(
            f"Row {row_idx}, feature {feature_name!r}: non-finite value {value!r} is not allowed"
        )
    return numeric


def _validate_feature_order_for_records(
    records: Sequence[Mapping[str, Any]],
    expected_order: tuple[str, ...],
    explicit_order: Sequence[str] | None = None,
) -> None:
    """Validate explicit column list and per-record feature key ordering."""
    if explicit_order is not None:
        actual_explicit = tuple(explicit_order)
        if actual_explicit != expected_order:
            raise FeatureOrderError(
                f"Expected feature order {expected_order}, got {actual_explicit}"
            )

    for row_idx, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise SchemaValidationError(
                f"Row {row_idx}: record must be a mapping, got {type(record).__name__}"
            )
        feature_keys = tuple(k for k in record.keys() if k not in METADATA_FIELDS)
        if feature_keys != expected_order:
            raise FeatureOrderError(
                f"Row {row_idx}: expected feature keys in order {expected_order}, "
                f"got {feature_keys}"
            )


# ── Fitted Transform-Only Public Interface ───────────────────────────────────


@dataclass(frozen=True)
class FittedNumericPreprocessor:
    """Immutable transform-only numeric preprocessor fitted on training rows.

    Notice that this public transform interface intentionally provides NO
    ``fit``, ``partial_fit``, or ``fit_transform`` method and freezes all
    learned statistics in immutable tuples. Fitting or mutating statistics on
    held-out input is therefore impossible through the public transform API.
    """

    feature_order: tuple[str, ...]
    valid_labels: frozenset[int]
    missing_value_policy: MissingValuePolicy
    means: tuple[float, ...]
    stds: tuple[float, ...]
    scales: tuple[float, ...]
    n_train_rows: int
    train_record_ids: frozenset[str]
    train_group_ids: frozenset[str]
    seed: int = FIXED_SEED

    def transform(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        feature_order: Sequence[str] | None = None,
        expected_split: str | None = None,
        allow_train_rows: bool = False,
    ) -> TransformedMatrix:
        """Standardize records using frozen training-only statistics.

        Parameters
        ----------
        records:
            Sequence of record mappings containing features in exact contract
            order (and optional metadata keys ``record_id``, ``group_id``,
            ``label``, ``split``).
        feature_order:
            Optional explicit column order declaration to verify against the
            fitted contract.
        expected_split:
            Optional partition label (``"train"``, ``"validation"``, or
            ``"test"``). When provided, every record with a ``"split"`` key
            must match it.
        allow_train_rows:
            Must be ``True`` only when transforming the training partition
            itself. Defaults to ``False`` so that held-out transform calls
            automatically reject any training ``record_id`` or ``group_id``
            contamination.
        """
        if not records:
            raise SchemaValidationError("Cannot transform an empty record sequence")

        if expected_split is not None and expected_split not in VALID_SPLITS:
            raise SchemaValidationError(
                f"Unknown expected_split {expected_split!r}; expected one of {sorted(VALID_SPLITS)}"
            )

        _validate_feature_order_for_records(
            records,
            expected_order=self.feature_order,
            explicit_order=feature_order,
        )

        transformed_rows: list[tuple[float, ...]] = []
        record_ids: list[str] = []
        seen_ids: set[str] = set()

        for row_idx, record in enumerate(records):
            record_id = _validate_non_empty_id(
                record.get("record_id"), "record_id", row_idx
            )
            if record_id in seen_ids:
                raise SchemaValidationError(
                    f"Duplicate record_id {record_id!r} in transform batch"
                )
            seen_ids.add(record_id)
            record_ids.append(record_id)

            group_id = (
                _validate_non_empty_id(record["group_id"], "group_id", row_idx)
                if "group_id" in record
                else None
            )

            if "split" in record and expected_split is not None:
                if record["split"] != expected_split:
                    raise ContaminationError(
                        f"Row {row_idx} ({record_id!r}): split={record['split']!r} "
                        f"contaminates expected_split={expected_split!r}"
                    )

            if not allow_train_rows:
                if record.get("split") == "train":
                    raise ContaminationError(
                        f"Row {row_idx} ({record_id!r}): training row passed into "
                        f"held-out transform call"
                    )
                if record_id in self.train_record_ids:
                    raise ContaminationError(
                        f"Row {row_idx}: record_id {record_id!r} overlaps training set"
                    )
                if group_id is not None and group_id in self.train_group_ids:
                    raise ContaminationError(
                        f"Row {row_idx}: group_id {group_id!r} overlaps training groups"
                    )

            if "label" in record:
                _validate_label(record["label"], self.valid_labels, row_idx)

            scaled_row: list[float] = []
            for col_idx, feature_name in enumerate(self.feature_order):
                parsed = _parse_numeric_feature(
                    record[feature_name],
                    feature_name=feature_name,
                    row_idx=row_idx,
                    policy=self.missing_value_policy,
                )
                if parsed is None:
                    # Only reachable under MissingValuePolicy.IMPUTE_TRAIN_MEAN:
                    # impute strictly from frozen training mean (no held-out leakage).
                    parsed = self.means[col_idx]
                standardized = float(
                    (parsed - self.means[col_idx]) / self.scales[col_idx]
                )
                scaled_row.append(standardized)
            transformed_rows.append(tuple(scaled_row))

        return TransformedMatrix(
            values=tuple(transformed_rows),
            shape=(len(transformed_rows), len(self.feature_order)),
            dtype=OUTPUT_DTYPE,
            feature_names=self.feature_order,
            record_ids=tuple(record_ids),
        )

    def transform_split(
        self,
        split_name: str,
        records: Sequence[Mapping[str, Any]],
        *,
        feature_order: Sequence[str] | None = None,
    ) -> TransformedMatrix:
        """Transform a named split with automatic train/held-out contamination checks."""
        if split_name not in VALID_SPLITS:
            raise SchemaValidationError(
                f"Unknown split_name {split_name!r}; expected one of {sorted(VALID_SPLITS)}"
            )
        return self.transform(
            records,
            feature_order=feature_order,
            expected_split=split_name,
            allow_train_rows=(split_name == "train"),
        )


# ── Training-Only Fitting Function ───────────────────────────────────────────


def fit_numeric_preprocessor(
    train_records: Sequence[Mapping[str, Any]],
    *,
    feature_order: Sequence[str] = EXPECTED_FEATURE_ORDER,
    valid_labels: Sequence[int] | frozenset[int] = VALID_LABELS,
    missing_value_policy: MissingValuePolicy | str = MissingValuePolicy.REJECT,
    held_out_records: Sequence[Mapping[str, Any]] | None = None,
    seed: int = FIXED_SEED,
) -> FittedNumericPreprocessor:
    """Fit numeric standardization strictly on validated training rows.

    Parameters
    ----------
    train_records:
        Training partition records. Any record tagged with a non-``"train"``
        split raises ``ContaminationError``.
    feature_order:
        Required ordered sequence of feature column names. Defaults to
        ``("x1", "x2")``.
    valid_labels:
        Allowed integer label set. Defaults to ``{0, 1}``.
    missing_value_policy:
        Either ``MissingValuePolicy.REJECT`` (default) or
        ``MissingValuePolicy.IMPUTE_TRAIN_MEAN``.
    held_out_records:
        Optional sequence of validation/test records used ONLY to assert
        disjoint ``record_id`` and ``group_id`` sets (feature values of
        held-out records are never inspected).
    seed:
        Fixed seed recorded in the fitted metadata for reproducibility.

    Returns
    -------
    FittedNumericPreprocessor
        Frozen, transform-only preprocessor containing training-only means,
        population standard deviations (ddof=0), and zero-variance-safe scales.
    """
    policy = _resolve_policy(missing_value_policy)
    expected_order = tuple(feature_order)
    if not expected_order or len(set(expected_order)) != len(expected_order):
        raise FeatureOrderError(
            f"feature_order must be a non-empty sequence of unique names, got {expected_order}"
        )

    allowed_labels = frozenset(valid_labels)
    if not allowed_labels or any(
        isinstance(lbl, bool) or not isinstance(lbl, int) for lbl in allowed_labels
    ):
        raise InvalidLabelError(
            f"valid_labels must be a non-empty set of ints, got {valid_labels!r}"
        )

    if not train_records:
        raise SchemaValidationError("Cannot fit preprocessor on empty train_records")

    _validate_feature_order_for_records(
        train_records,
        expected_order=expected_order,
    )

    train_record_ids: set[str] = set()
    train_group_ids: set[str] = set()
    column_values: list[list[float | None]] = [[] for _ in expected_order]

    for row_idx, record in enumerate(train_records):
        if "split" in record and record["split"] != "train":
            raise ContaminationError(
                f"Row {row_idx}: cannot fit on non-train record with split={record['split']!r}"
            )

        record_id = _validate_non_empty_id(
            record.get("record_id"), "record_id", row_idx
        )
        if record_id in train_record_ids:
            raise SchemaValidationError(
                f"Duplicate record_id {record_id!r} in training records"
            )
        train_record_ids.add(record_id)

        group_id = _validate_non_empty_id(
            record.get("group_id"), "group_id", row_idx
        )
        train_group_ids.add(group_id)

        if "label" not in record:
            raise InvalidLabelError(f"Row {row_idx}: missing required 'label' field")
        _validate_label(record["label"], allowed_labels, row_idx)

        for col_idx, feature_name in enumerate(expected_order):
            parsed = _parse_numeric_feature(
                record[feature_name],
                feature_name=feature_name,
                row_idx=row_idx,
                policy=policy,
            )
            column_values[col_idx].append(parsed)

    if held_out_records is not None:
        for h_idx, h_record in enumerate(held_out_records):
            if not isinstance(h_record, Mapping):
                raise SchemaValidationError(
                    f"Held-out row {h_idx}: record must be a mapping"
                )
            h_rec_id = _validate_non_empty_id(
                h_record.get("record_id"), "record_id", h_idx
            )
            h_grp_id = _validate_non_empty_id(
                h_record.get("group_id"), "group_id", h_idx
            )
            if h_rec_id in train_record_ids:
                raise ContaminationError(
                    f"Contamination detected: record_id {h_rec_id!r} appears in both "
                    "train and held-out partitions"
                )
            if h_grp_id in train_group_ids:
                raise ContaminationError(
                    f"Contamination detected: group_id {h_grp_id!r} appears in both "
                    "train and held-out partitions"
                )

    means: list[float] = []
    stds: list[float] = []
    scales: list[float] = []

    for col_idx, feature_name in enumerate(expected_order):
        raw_col = column_values[col_idx]
        observed = [v for v in raw_col if v is not None]
        if not observed:
            raise MissingValueError(
                f"Feature {feature_name!r} has no non-missing values in training data"
            )
        mean_val = float(sum(observed) / len(observed))
        filled_col = [mean_val if v is None else v for v in raw_col]
        variance = sum((v - mean_val) ** 2 for v in filled_col) / len(filled_col)
        std_val = float(math.sqrt(variance))
        # Match scikit-learn StandardScaler zero-variance convention: scale=1.0
        scale_val = std_val if std_val > 0.0 else 1.0
        means.append(mean_val)
        stds.append(std_val)
        scales.append(scale_val)

    return FittedNumericPreprocessor(
        feature_order=expected_order,
        valid_labels=allowed_labels,
        missing_value_policy=policy,
        means=tuple(means),
        stds=tuple(stds),
        scales=tuple(scales),
        n_train_rows=len(train_records),
        train_record_ids=frozenset(train_record_ids),
        train_group_ids=frozenset(train_group_ids),
        seed=seed,
    )


# ── Fixture Loading & Partition Validation ───────────────────────────────────


def load_fixture(path: Path | str | None = None) -> dict[str, Any]:
    """Load the synthetic fixture JSON from disk."""
    fixture_path = Path(path) if path is not None else DEFAULT_FIXTURE_PATH
    with fixture_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not data.get("is_synthetic", False):
        raise SchemaValidationError(
            "Fixture must explicitly declare is_synthetic=true for pipeline verification"
        )
    return data


def load_schema(path: Path | str | None = None) -> dict[str, Any]:
    """Load the data contract schema JSON from disk."""
    schema_path = Path(path) if path is not None else DEFAULT_SCHEMA_PATH
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_and_partition_fixture(
    records: Sequence[Mapping[str, Any]],
    *,
    feature_order: Sequence[str] = EXPECTED_FEATURE_ORDER,
    valid_labels: Sequence[int] | frozenset[int] = VALID_LABELS,
    missing_value_policy: MissingValuePolicy | str = MissingValuePolicy.REJECT,
) -> SplitPartitions:
    """Validate schema and partition records into disjoint train/val/test splits.

    Rejects:
      - Missing required fields or invalid split labels.
      - Wrong feature order or missing values (according to policy).
      - Invalid class labels outside ``valid_labels``.
      - Duplicate ``record_id`` values anywhere in the dataset.
      - Any ``group_id`` shared across two different partitions (group leakage).
      - Empty or unstratified partitions (every split must include every class).
    """
    policy = _resolve_policy(missing_value_policy)
    expected_order = tuple(feature_order)
    allowed_labels = frozenset(valid_labels)

    if not records:
        raise SchemaValidationError("Fixture records must not be empty")

    _validate_feature_order_for_records(records, expected_order=expected_order)

    buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    record_to_split: dict[str, str] = {}
    group_to_split: dict[str, str] = {}

    for row_idx, record in enumerate(records):
        record_id = _validate_non_empty_id(
            record.get("record_id"), "record_id", row_idx
        )
        group_id = _validate_non_empty_id(
            record.get("group_id"), "group_id", row_idx
        )
        split = record.get("split")
        if split not in VALID_SPLITS:
            raise SchemaValidationError(
                f"Row {row_idx} ({record_id!r}): split must be one of "
                f"{sorted(VALID_SPLITS)}, got {split!r}"
            )

        if record_id in record_to_split:
            prev_split = record_to_split[record_id]
            if prev_split != split:
                raise ContaminationError(
                    f"Contamination detected: record_id {record_id!r} appears in "
                    f"both {prev_split!r} and {split!r}"
                )
            raise SchemaValidationError(
                f"Duplicate record_id {record_id!r} within split {split!r}"
            )
        record_to_split[record_id] = split

        if group_id in group_to_split and group_to_split[group_id] != split:
            raise ContaminationError(
                f"Contamination detected: group_id {group_id!r} spans both "
                f"{group_to_split[group_id]!r} and {split!r}"
            )
        group_to_split[group_id] = split

        if "label" not in record:
            raise InvalidLabelError(f"Row {row_idx}: missing required 'label' field")
        _validate_label(record["label"], allowed_labels, row_idx)

        for feature_name in expected_order:
            _parse_numeric_feature(
                record[feature_name],
                feature_name=feature_name,
                row_idx=row_idx,
                policy=policy,
            )

        buckets[split].append(dict(record))

    for split_name, rows in buckets.items():
        if not rows:
            raise SchemaValidationError(
                f"Partition {split_name!r} must not be empty"
            )
        present_labels = {row["label"] for row in rows}
        if present_labels != allowed_labels:
            raise SchemaValidationError(
                f"Partition {split_name!r} is not stratified over {sorted(allowed_labels)}: "
                f"observed {sorted(present_labels)}"
            )

    return SplitPartitions(
        train=tuple(buckets["train"]),
        validation=tuple(buckets["validation"]),
        test=tuple(buckets["test"]),
    )


def run_verification_pipeline(
    fixture_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute the full offline synthetic pipeline verification and return results."""
    fixture_doc = load_fixture(fixture_path)
    partitions = validate_and_partition_fixture(
        fixture_doc["records"],
        feature_order=tuple(fixture_doc["feature_order"]),
        valid_labels=frozenset(fixture_doc["valid_labels"]),
        missing_value_policy=MissingValuePolicy.REJECT,
    )
    fitted = fit_numeric_preprocessor(
        partitions.train,
        feature_order=tuple(fixture_doc["feature_order"]),
        valid_labels=frozenset(fixture_doc["valid_labels"]),
        missing_value_policy=MissingValuePolicy.REJECT,
        held_out_records=partitions.validation + partitions.test,
        seed=int(fixture_doc.get("seed", FIXED_SEED)),
    )

    train_out = fitted.transform_split("train", partitions.train)
    val_out = fitted.transform_split("validation", partitions.validation)
    test_out = fitted.transform_split("test", partitions.test)

    return {
        "task_id": TASK_ID,
        "project_section": PROJECT_SECTION,
        "verification_only": True,
        "synthetic_disclaimer": fixture_doc["disclaimer"],
        "seed": fitted.seed,
        "feature_order": list(fitted.feature_order),
        "valid_labels": sorted(fitted.valid_labels),
        "missing_value_policy": fitted.missing_value_policy.value,
        "fitted_on_train_only": {
            "n_train_rows": fitted.n_train_rows,
            "train_record_ids": sorted(fitted.train_record_ids),
            "train_group_ids": sorted(fitted.train_group_ids),
            "means": list(fitted.means),
            "stds": list(fitted.stds),
            "scales": list(fitted.scales),
        },
        "splits": {
            "train": {
                "record_ids": list(train_out.record_ids),
                "shape": list(train_out.shape),
                "dtype": train_out.dtype,
                "values": train_out.to_lists(),
            },
            "validation": {
                "record_ids": list(val_out.record_ids),
                "shape": list(val_out.shape),
                "dtype": val_out.dtype,
                "values": val_out.to_lists(),
            },
            "test": {
                "record_ids": list(test_out.record_ids),
                "shape": list(test_out.shape),
                "dtype": test_out.dtype,
                "values": test_out.to_lists(),
            },
        },
    }
