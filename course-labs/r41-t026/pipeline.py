"""Bounded CPU regression smoke-test pipeline for R41-T026.

Task: R41-T026 · 6 of 10
Project/section: Building energy efficiency – regression
Student: Sharon Robinson
Repository: S-AI-Projects/ML-CaPsule

SYNTHETIC DATA NOTICE
=====================
All default fixture records and verification outputs produced by this module
use a six-row deterministic formula fixture (``target = 2 * x1 + 1``).
They are strictly for bounded CPU smoke-test pipeline verification — NOT
observations from the UCI Energy Efficiency dataset and NOT evidence of
real-world or clinical performance.

Bounded Smoke Contract
----------------------
- Sample bound: at most ``MAX_SAMPLES = 32`` samples.
- Training budget: at most ``MAX_EPOCHS = 1`` epoch and at most
  ``MAX_UPDATE_STEPS = 20`` gradient update steps.
- Fixed seed: ``FIXED_SEED = 42`` for deterministic reproducibility.
- Offline & CPU-only: Python 3 standard library only; requires no GPU,
  credentials, network access, or dataset downloads.
- Interface assertions: verifies finite outputs and MSE loss, exact prediction
  shape ``(n_samples,)`` with ``dtype="float64"``, recorded ``step_count``,
  and explicit train/validation/test separation — asserting interface
  correctness rather than useful predictive accuracy.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# ── Constants & Hard Smoke Bounds ────────────────────────────────────────────

TASK_ID: str = "R41-T026"
PROJECT_SECTION: str = "Building energy efficiency – regression"
FIXTURE_KIND: str = "deterministic_formula_fixture"
FIXED_SEED: int = 42
MAX_SAMPLES: int = 32
MAX_EPOCHS: int = 1
MAX_UPDATE_STEPS: int = 20
DEFAULT_LEARNING_RATE: float = 0.05
EXPECTED_FEATURE_ORDER: tuple[str, ...] = ("x1", "x2")
REQUIRED_RECORD_FIELDS: tuple[str, ...] = ("record_id", "x1", "x2", "target")
VALID_SPLITS: frozenset[str] = frozenset({"train", "validation", "test"})
OUTPUT_DTYPE: str = "float64"

LAB_DIR: Path = Path(__file__).resolve().parent
DEFAULT_FIXTURE_PATH: Path = LAB_DIR / "fixture.json"
DEFAULT_SCHEMA_PATH: Path = LAB_DIR / "schema.json"


# ── Exceptions ───────────────────────────────────────────────────────────────


class SmokePipelineError(ValueError):
    """Base exception for R41-T026 smoke-test contract violations."""


class SchemaValidationError(SmokePipelineError):
    """Raised when record fields, types, or numeric values violate the schema."""


class SmokeBoundError(SmokePipelineError):
    """Raised when sample count, epochs, step count, or lr exceeds smoke bounds."""


class SplitContaminationError(SmokePipelineError):
    """Raised when train/validation/test partitions overlap or leak into fit."""


class ModelNotFittedError(RuntimeError):
    """Raised when predict or evaluate is called before fitting."""


# ── Immutable Data Structures ────────────────────────────────────────────────


@dataclass(frozen=True)
class PredictionOutput:
    """Immutable 1-D regression predictions with explicit shape and float64 dtype."""

    values: tuple[float, ...]
    shape: tuple[int]
    dtype: str
    record_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.dtype != OUTPUT_DTYPE:
            raise SchemaValidationError(
                f"Expected dtype {OUTPUT_DTYPE!r}, got {self.dtype!r}"
            )
        n_rows = len(self.values)
        if self.shape != (n_rows,):
            raise SchemaValidationError(
                f"Shape mismatch: declared {self.shape}, actual {(n_rows,)}"
            )
        if len(self.record_ids) != n_rows:
            raise SchemaValidationError(
                f"record_ids length {len(self.record_ids)} != prediction count {n_rows}"
            )
        for idx, val in enumerate(self.values):
            if isinstance(val, bool) or not isinstance(val, float):
                raise SchemaValidationError(
                    f"Prediction at index {idx} must be float64, got {type(val).__name__}"
                )
            if not math.isfinite(val):
                raise SchemaValidationError(
                    f"Non-finite prediction at index {idx}: {val!r}"
                )

    def to_list(self) -> list[float]:
        """Return a mutable list copy of prediction values."""
        return list(self.values)


@dataclass(frozen=True)
class SplitEvaluation:
    """Immutable evaluation summary for a single named partition."""

    split_name: str
    predictions: PredictionOutput
    targets: tuple[float, ...]
    mse_loss: float
    mae_loss: float


@dataclass(frozen=True)
class SplitPartitions:
    """Validated, pairwise-disjoint train, validation, and test partitions."""

    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    test: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SmokeRunReport:
    """Structured training and inference smoke-run report."""

    model_name: str
    model_kind: str
    is_naive_baseline: bool
    asserts_interface_only: bool
    seed: int
    epochs_completed: int
    step_count: int
    step_losses: tuple[float, ...]
    parameters: dict[str, float]
    train_eval: SplitEvaluation
    validation_eval: SplitEvaluation
    test_eval: SplitEvaluation


# ── Validation Helpers ───────────────────────────────────────────────────────


def _validate_non_empty_id(value: Any, field_name: str, row_idx: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(
            f"Row {row_idx}: {field_name!r} must be a non-empty string, got {value!r}"
        )
    return value


def _validate_finite_number(value: Any, field_name: str, row_idx: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(
            f"Row {row_idx}: {field_name!r} must be a finite number, "
            f"got {value!r} ({type(value).__name__})"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise SchemaValidationError(
            f"Row {row_idx}: {field_name!r} must be finite, got {value!r}"
        )
    return numeric


def _validate_record_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    require_target: bool = True,
    require_train_split: bool = False,
) -> tuple[list[str], list[tuple[float, float]], list[float]]:
    """Validate bounded sample count, schema fields, and finite numerics."""
    if not records:
        raise SchemaValidationError("Record sequence must not be empty")
    if len(records) > MAX_SAMPLES:
        raise SmokeBoundError(
            f"Sample count {len(records)} exceeds smoke-test bound MAX_SAMPLES={MAX_SAMPLES}"
        )

    record_ids: list[str] = []
    seen_ids: set[str] = set()
    features: list[tuple[float, float]] = []
    targets: list[float] = []

    for row_idx, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise SchemaValidationError(
                f"Row {row_idx}: record must be a mapping, got {type(rec).__name__}"
            )
        rec_id = _validate_non_empty_id(rec.get("record_id"), "record_id", row_idx)
        if rec_id in seen_ids:
            raise SchemaValidationError(f"Duplicate record_id {rec_id!r} in batch")
        seen_ids.add(rec_id)

        if require_train_split and "split" in rec and rec["split"] != "train":
            raise SplitContaminationError(
                f"Row {row_idx} ({rec_id!r}): cannot fit on non-train split={rec['split']!r}"
            )

        for feat in EXPECTED_FEATURE_ORDER:
            if feat not in rec:
                raise SchemaValidationError(
                    f"Row {row_idx} ({rec_id!r}): missing required feature {feat!r}"
                )
        x1 = _validate_finite_number(rec["x1"], "x1", row_idx)
        x2 = _validate_finite_number(rec["x2"], "x2", row_idx)

        record_ids.append(rec_id)
        features.append((x1, x2))

        if require_target:
            if "target" not in rec:
                raise SchemaValidationError(
                    f"Row {row_idx} ({rec_id!r}): missing required field 'target'"
                )
            targets.append(_validate_finite_number(rec["target"], "target", row_idx))

    return record_ids, features, targets


def compute_regression_losses(
    predictions: PredictionOutput,
    targets: Sequence[float],
) -> tuple[float, float]:
    """Compute finite (MSE, MAE) losses over aligned predictions and targets."""
    if len(predictions.values) != len(targets) or not targets:
        raise SchemaValidationError(
            f"Prediction count {len(predictions.values)} != target count {len(targets)}"
        )
    sq_err = 0.0
    abs_err = 0.0
    for pred, target in zip(predictions.values, targets):
        diff = pred - target
        sq_err += diff * diff
        abs_err += abs(diff)
    n = float(len(targets))
    mse = sq_err / n
    mae = abs_err / n
    if not (math.isfinite(mse) and math.isfinite(mae)):
        raise SchemaValidationError(f"Computed non-finite loss: mse={mse!r}, mae={mae!r}")
    return mse, mae


# ── Bounded CPU Smoke Models ─────────────────────────────────────────────────


class BoundedLinearSGDSmokeRegressor:
    """Tiny CPU-only linear SGD regressor bounded to <=1 epoch and <=20 steps.

    Explicitly intended as a deterministic pipeline smoke test verifying
    training/inference interface correctness (finite losses/outputs, prediction
    shapes, step-count accounting, and reproducibility) — NOT tuned for useful
    real-world predictive accuracy.
    """

    model_name: str = "BoundedLinearSGDSmokeRegressor"
    model_kind: str = "bounded_cpu_linear_sgd_smoke_model"
    is_naive_baseline: bool = False
    asserts_interface_only: bool = True

    def __init__(
        self,
        *,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        epochs: int = MAX_EPOCHS,
        max_steps: int = MAX_UPDATE_STEPS,
        seed: int = FIXED_SEED,
    ) -> None:
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            raise SmokeBoundError(f"learning_rate must be numeric, got {learning_rate!r}")
        if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
            raise SmokeBoundError(
                f"learning_rate must be positive and finite, got {learning_rate!r}"
            )
        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1 or epochs > MAX_EPOCHS:
            raise SmokeBoundError(
                f"epochs must be an integer in [1, {MAX_EPOCHS}], got {epochs!r}"
            )
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < 1
            or max_steps > MAX_UPDATE_STEPS
        ):
            raise SmokeBoundError(
                f"max_steps must be an integer in [1, {MAX_UPDATE_STEPS}], got {max_steps!r}"
            )
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise SmokeBoundError(f"seed must be an integer, got {seed!r}")

        self.learning_rate: float = float(learning_rate)
        self.epochs: int = epochs
        self.max_steps: int = max_steps
        self.seed: int = seed

        self.w1_: float | None = None
        self.w2_: float | None = None
        self.epochs_completed_: int = 0
        self.step_count_: int = 0
        self.step_losses_: tuple[float, ...] = ()
        self.train_record_ids_: frozenset[str] = frozenset()
        self._is_fitted: bool = False

    def fit(self, train_records: Sequence[Mapping[str, Any]]) -> "BoundedLinearSGDSmokeRegressor":
        """Fit on training rows only for at most 1 epoch and <= 20 update steps."""
        record_ids, features, targets = _validate_record_batch(
            train_records,
            require_target=True,
            require_train_split=True,
        )

        rng = random.Random(self.seed)
        w1 = 0.0
        w2 = 0.0
        step_losses: list[float] = []
        steps = 0
        epochs_done = 0

        for _ in range(self.epochs):
            order = list(range(len(features)))
            rng.shuffle(order)
            for idx in order:
                if steps >= self.max_steps:
                    break
                x1, x2 = features[idx]
                y = targets[idx]
                y_hat = w1 * x1 + w2 * x2
                err = y_hat - y
                loss = err * err
                if not math.isfinite(loss):
                    raise SmokePipelineError(f"Non-finite step loss encountered: {loss!r}")
                step_losses.append(float(loss))

                grad_w1 = 2.0 * err * x1
                grad_w2 = 2.0 * err * x2
                w1 = float(w1 - self.learning_rate * grad_w1)
                w2 = float(w2 - self.learning_rate * grad_w2)
                if not (math.isfinite(w1) and math.isfinite(w2)):
                    raise SmokePipelineError("Non-finite weights encountered during SGD")
                steps += 1
            epochs_done += 1

        self.w1_ = w1
        self.w2_ = w2
        self.epochs_completed_ = epochs_done
        self.step_count_ = steps
        self.step_losses_ = tuple(step_losses)
        self.train_record_ids_ = frozenset(record_ids)
        self._is_fitted = True
        return self

    def predict(self, records: Sequence[Mapping[str, Any]]) -> PredictionOutput:
        """Produce finite float64 predictions with shape (n_samples,)."""
        if not self._is_fitted or self.w1_ is None or self.w2_ is None:
            raise ModelNotFittedError("Call .fit() on training records before .predict()")
        record_ids, features, _ = _validate_record_batch(
            records,
            require_target=False,
            require_train_split=False,
        )
        preds = tuple(float(self.w1_ * x1 + self.w2_ * x2) for x1, x2 in features)
        return PredictionOutput(
            values=preds,
            shape=(len(preds),),
            dtype=OUTPUT_DTYPE,
            record_ids=tuple(record_ids),
        )

    def evaluate(
        self,
        split_name: str,
        records: Sequence[Mapping[str, Any]],
    ) -> SplitEvaluation:
        """Evaluate predictions against targets for a named partition."""
        if split_name not in VALID_SPLITS:
            raise SchemaValidationError(
                f"Unknown split_name {split_name!r}; expected one of {sorted(VALID_SPLITS)}"
            )
        record_ids, _, targets = _validate_record_batch(
            records,
            require_target=True,
            require_train_split=(split_name == "train"),
        )
        if split_name != "train":
            overlap = self.train_record_ids_ & set(record_ids)
            if overlap:
                raise SplitContaminationError(
                    f"Held-out split {split_name!r} contaminated with train record_ids {sorted(overlap)}"
                )
        preds = self.predict(records)
        mse, mae = compute_regression_losses(preds, targets)
        return SplitEvaluation(
            split_name=split_name,
            predictions=preds,
            targets=tuple(targets),
            mse_loss=mse,
            mae_loss=mae,
        )


class NaiveMeanBaseline:
    """Explicitly labelled constant/naive baseline predicting the training mean.

    ``is_naive_baseline = True`` and ``model_kind = "constant_naive_baseline"``
    make the constant prediction behaviour explicit so it can never be mistaken
    for a trained predictive energy model.
    """

    model_name: str = "NaiveMeanBaseline"
    model_kind: str = "constant_naive_baseline"
    is_naive_baseline: bool = True
    asserts_interface_only: bool = True

    def __init__(self, *, seed: int = FIXED_SEED) -> None:
        self.seed: int = seed
        self.train_mean_: float | None = None
        self.epochs_completed_: int = 0
        self.step_count_: int = 0
        self.step_losses_: tuple[float, ...] = ()
        self.train_record_ids_: frozenset[str] = frozenset()
        self._is_fitted: bool = False

    def fit(self, train_records: Sequence[Mapping[str, Any]]) -> "NaiveMeanBaseline":
        """Fit the constant mean prediction strictly from training targets."""
        record_ids, _, targets = _validate_record_batch(
            train_records,
            require_target=True,
            require_train_split=True,
        )
        mean_val = float(sum(targets) / len(targets))
        train_mse = float(sum((mean_val - y) ** 2 for y in targets) / len(targets))

        self.train_mean_ = mean_val
        self.epochs_completed_ = 1
        self.step_count_ = 1
        self.step_losses_ = (train_mse,)
        self.train_record_ids_ = frozenset(record_ids)
        self._is_fitted = True
        return self

    def predict(self, records: Sequence[Mapping[str, Any]]) -> PredictionOutput:
        """Predict the constant training mean with shape (n_samples,)."""
        if not self._is_fitted or self.train_mean_ is None:
            raise ModelNotFittedError("Call .fit() on training records before .predict()")
        record_ids, _, _ = _validate_record_batch(
            records,
            require_target=False,
            require_train_split=False,
        )
        preds = tuple(float(self.train_mean_) for _ in record_ids)
        return PredictionOutput(
            values=preds,
            shape=(len(preds),),
            dtype=OUTPUT_DTYPE,
            record_ids=tuple(record_ids),
        )

    def evaluate(
        self,
        split_name: str,
        records: Sequence[Mapping[str, Any]],
    ) -> SplitEvaluation:
        """Evaluate constant baseline predictions on a named partition."""
        if split_name not in VALID_SPLITS:
            raise SchemaValidationError(
                f"Unknown split_name {split_name!r}; expected one of {sorted(VALID_SPLITS)}"
            )
        record_ids, _, targets = _validate_record_batch(
            records,
            require_target=True,
            require_train_split=(split_name == "train"),
        )
        if split_name != "train":
            overlap = self.train_record_ids_ & set(record_ids)
            if overlap:
                raise SplitContaminationError(
                    f"Held-out split {split_name!r} contaminated with train record_ids {sorted(overlap)}"
                )
        preds = self.predict(records)
        mse, mae = compute_regression_losses(preds, targets)
        return SplitEvaluation(
            split_name=split_name,
            predictions=preds,
            targets=tuple(targets),
            mse_loss=mse,
            mae_loss=mae,
        )


# ── Fixture Loading & Partitioning ───────────────────────────────────────────


def load_fixture(path: Path | str | None = None) -> dict[str, Any]:
    """Load and verify the synthetic deterministic formula fixture JSON."""
    fixture_path = Path(path) if path is not None else DEFAULT_FIXTURE_PATH
    with fixture_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not data.get("is_synthetic", False):
        raise SchemaValidationError("Fixture must explicitly declare is_synthetic=true")
    if data.get("fixture_kind") != FIXTURE_KIND:
        raise SchemaValidationError(
            f"Fixture must declare fixture_kind={FIXTURE_KIND!r}, got {data.get('fixture_kind')!r}"
        )
    return data


def load_schema(path: Path | str | None = None) -> dict[str, Any]:
    """Load the R41-T026 schema JSON."""
    schema_path = Path(path) if path is not None else DEFAULT_SCHEMA_PATH
    with schema_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_and_partition_fixture(
    records: Sequence[Mapping[str, Any]],
    *,
    verify_formula: bool = True,
) -> SplitPartitions:
    """Validate schema, sample bound (<=32), optional formula, and disjoint splits."""
    if not records:
        raise SchemaValidationError("Fixture records must not be empty")
    if len(records) > MAX_SAMPLES:
        raise SmokeBoundError(
            f"Fixture size {len(records)} exceeds smoke-test bound MAX_SAMPLES={MAX_SAMPLES}"
        )

    buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    record_to_split: dict[str, str] = {}

    for row_idx, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise SchemaValidationError(
                f"Row {row_idx}: record must be a mapping, got {type(rec).__name__}"
            )
        rec_id = _validate_non_empty_id(rec.get("record_id"), "record_id", row_idx)
        split = rec.get("split")
        if split not in VALID_SPLITS:
            raise SchemaValidationError(
                f"Row {row_idx} ({rec_id!r}): invalid split {split!r}; expected {sorted(VALID_SPLITS)}"
            )
        if rec_id in record_to_split:
            prev = record_to_split[rec_id]
            if prev != split:
                raise SplitContaminationError(
                    f"Contamination detected: record_id {rec_id!r} appears in both {prev!r} and {split!r}"
                )
            raise SchemaValidationError(
                f"Duplicate record_id {rec_id!r} within split {split!r}"
            )
        record_to_split[rec_id] = split

        for field in REQUIRED_RECORD_FIELDS:
            if field not in rec:
                raise SchemaValidationError(
                    f"Row {row_idx} ({rec_id!r}): missing required field {field!r}"
                )

        x1 = _validate_finite_number(rec["x1"], "x1", row_idx)
        _validate_finite_number(rec["x2"], "x2", row_idx)
        target = _validate_finite_number(rec["target"], "target", row_idx)

        if verify_formula:
            expected_target = 2.0 * x1 + 1.0
            if not math.isclose(target, expected_target, rel_tol=0.0, abs_tol=1e-12):
                raise SchemaValidationError(
                    f"Row {row_idx} ({rec_id!r}): target={target} violates formula 2*x1+1={expected_target}"
                )

        buckets[split].append(dict(rec))

    for split_name, rows in buckets.items():
        if not rows:
            raise SchemaValidationError(f"Partition {split_name!r} must not be empty")

    return SplitPartitions(
        train=tuple(buckets["train"]),
        validation=tuple(buckets["validation"]),
        test=tuple(buckets["test"]),
    )


def _eval_to_dict(ev: SplitEvaluation) -> dict[str, Any]:
    return {
        "split": ev.split_name,
        "record_ids": list(ev.predictions.record_ids),
        "prediction_shape": list(ev.predictions.shape),
        "dtype": ev.predictions.dtype,
        "predictions": ev.predictions.to_list(),
        "targets": list(ev.targets),
        "mse_loss": ev.mse_loss,
        "mae_loss": ev.mae_loss,
    }


def run_smoke_pipeline(fixture_path: Path | str | None = None) -> dict[str, Any]:
    """Run the bounded CPU regression smoke test and return structured verification data."""
    fixture_doc = load_fixture(fixture_path)
    partitions = validate_and_partition_fixture(
        fixture_doc["records"],
        verify_formula=True,
    )
    seed = int(fixture_doc.get("seed", FIXED_SEED))

    # 1. Bounded 1-epoch / <=20-step CPU SGD smoke model
    sgd_model = BoundedLinearSGDSmokeRegressor(
        learning_rate=DEFAULT_LEARNING_RATE,
        epochs=MAX_EPOCHS,
        max_steps=MAX_UPDATE_STEPS,
        seed=seed,
    ).fit(partitions.train)
    sgd_report = SmokeRunReport(
        model_name=sgd_model.model_name,
        model_kind=sgd_model.model_kind,
        is_naive_baseline=sgd_model.is_naive_baseline,
        asserts_interface_only=sgd_model.asserts_interface_only,
        seed=seed,
        epochs_completed=sgd_model.epochs_completed_,
        step_count=sgd_model.step_count_,
        step_losses=sgd_model.step_losses_,
        parameters={"w1": float(sgd_model.w1_), "w2": float(sgd_model.w2_)},
        train_eval=sgd_model.evaluate("train", partitions.train),
        validation_eval=sgd_model.evaluate("validation", partitions.validation),
        test_eval=sgd_model.evaluate("test", partitions.test),
    )

    # 2. Explicitly labelled constant/naive training-mean baseline
    naive_model = NaiveMeanBaseline(seed=seed).fit(partitions.train)
    naive_report = SmokeRunReport(
        model_name=naive_model.model_name,
        model_kind=naive_model.model_kind,
        is_naive_baseline=naive_model.is_naive_baseline,
        asserts_interface_only=naive_model.asserts_interface_only,
        seed=seed,
        epochs_completed=naive_model.epochs_completed_,
        step_count=naive_model.step_count_,
        step_losses=naive_model.step_losses_,
        parameters={"train_mean": float(naive_model.train_mean_)},
        train_eval=naive_model.evaluate("train", partitions.train),
        validation_eval=naive_model.evaluate("validation", partitions.validation),
        test_eval=naive_model.evaluate("test", partitions.test),
    )

    def _report_to_dict(rep: SmokeRunReport) -> dict[str, Any]:
        return {
            "model_name": rep.model_name,
            "model_kind": rep.model_kind,
            "is_naive_baseline": rep.is_naive_baseline,
            "asserts_interface_only": rep.asserts_interface_only,
            "seed": rep.seed,
            "epochs_completed": rep.epochs_completed,
            "step_count": rep.step_count,
            "step_losses": list(rep.step_losses),
            "parameters": rep.parameters,
            "splits": {
                "train": _eval_to_dict(rep.train_eval),
                "validation": _eval_to_dict(rep.validation_eval),
                "test": _eval_to_dict(rep.test_eval),
            },
        }

    return {
        "task_id": TASK_ID,
        "project_section": PROJECT_SECTION,
        "fixture_kind": FIXTURE_KIND,
        "verification_only": True,
        "synthetic_disclaimer": fixture_doc["disclaimer"],
        "smoke_bounds": {
            "max_samples": MAX_SAMPLES,
            "observed_samples": len(fixture_doc["records"]),
            "max_epochs": MAX_EPOCHS,
            "max_update_steps": MAX_UPDATE_STEPS,
            "fixed_seed": seed,
        },
        "models": {
            "bounded_sgd_smoke": _report_to_dict(sgd_report),
            "constant_naive_baseline": _report_to_dict(naive_report),
        },
    }
