"""Saved-model JSON interface and chronological trend forecast for R41-T030.

Task: R41-T030 · 10 of 10
Project/section: Global CO2 emissions – trend forecast
Student: Sharon Robinson
Repository: S-AI-Projects/ML-CaPsule

SYNTHETIC DATA NOTICE
=====================
All default fixture records (d1..d10) and verification forecasts produced by
this module use ten synthetic consecutive daily values strictly for pipeline
verification. They are NOT real Global CO2 emissions observations and must
NOT be cited as evidence of real-world or clinical performance.

Design Guarantees
-----------------
1. Chronological train/validation/test separation (no shuffling):
   - First 6 consecutive days (``d1..d6``) -> ``train``
   - Next 2 consecutive days (``d7..d8``) -> ``validation``
   - Final 2 consecutive days (``d9..d10``) -> ``test``
   - Out-of-order or non-consecutive daily timestamps are rejected with
     ``ChronologyError``.
2. Safe JSON persistence (never loads pickle):
   - ``save_artifact`` / ``load_artifact`` serialize and validate only safe
     JSON parameters (``intercept``, ``slope``, ``origin_timestamp``,
     ``schema_version``, ``input_schema``).
   - ``.pkl`` / ``.pickle`` files are explicitly rejected with
     ``UnsafeArtifactFormatError``.
   - No training observations or credentials are stored in or needed by the
     reloaded artifact at inference time.
3. Strict schema/version validation & equivalence:
   - Before-save and after-reload predictions agree within explicit tolerance
     ``RELOAD_TOLERANCE = 1e-12``.
   - Version mismatches (``SchemaVersionMismatchError``), unknown/malformed
     artifact fields (``ArtifactValidationError``), and invalid inference
     payloads (``InferenceSchemaError``) fail clearly.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

# ── Constants ────────────────────────────────────────────────────────────────

TASK_ID: str = "R41-T030"
PROJECT_SECTION: str = "Global CO2 emissions – trend forecast"
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0.0"})
DEFAULT_SCHEMA_VERSION: str = "1.0.0"
FIXED_SEED: int = 42
RELOAD_TOLERANCE: float = 1e-12
TRAIN_COUNT: int = 6
VAL_COUNT: int = 2
TEST_COUNT: int = 2
REQUIRED_FIXTURE_FIELDS: tuple[str, ...] = ("record_id", "timestamp", "value")
REQUIRED_INFERENCE_FIELDS: tuple[str, ...] = ("record_id", "timestamp")
ALLOWED_ARTIFACT_TOP_KEYS: frozenset[str] = frozenset(
    {
        "artifact_format",
        "schema_version",
        "task_id",
        "model_type",
        "seed",
        "origin_timestamp",
        "train_end_timestamp",
        "n_train_observations",
        "input_schema",
        "parameters",
    }
)
ALLOWED_PARAM_KEYS: frozenset[str] = frozenset({"intercept", "slope"})
ALLOWED_INPUT_SCHEMA_KEYS: frozenset[str] = frozenset(
    {"required_fields", "timestamp_format"}
)
FORBIDDEN_EXTENSIONS: frozenset[str] = frozenset({".pkl", ".pickle", ".joblib", ".dill"})

LAB_DIR: Path = Path(__file__).resolve().parent
DEFAULT_FIXTURE_PATH: Path = LAB_DIR / "fixture.json"
DEFAULT_SCHEMA_PATH: Path = LAB_DIR / "schema.json"
DEFAULT_ARTIFACT_PATH: Path = LAB_DIR / "saved_model.json"


# ── Exceptions ───────────────────────────────────────────────────────────────


class SavedModelError(ValueError):
    """Base exception for R41-T030 saved-model and forecast contract errors."""


class ChronologyError(SavedModelError):
    """Raised when timestamps are not strictly consecutive daily order."""


class UnsafeArtifactFormatError(SavedModelError):
    """Raised when a pickle or other unsafe artifact format is provided."""


class SchemaVersionMismatchError(SavedModelError):
    """Raised when artifact schema_version is missing or unsupported."""


class ArtifactValidationError(SavedModelError):
    """Raised when saved artifact structure or parameters are invalid."""


class InferenceSchemaError(SavedModelError):
    """Raised when inference records violate the expected schema."""


# ── Immutable Data Structures ────────────────────────────────────────────────


@dataclass(frozen=True)
class TimeSeriesSplits:
    """Strict chronological train/validation/test partitions (no shuffling)."""

    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    test: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ForecastPrediction:
    """Single validated forecast output."""

    record_id: str
    timestamp: str
    day_index: int
    predicted_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "timestamp": self.timestamp,
            "day_index": self.day_index,
            "predicted_value": self.predicted_value,
        }


@dataclass(frozen=True)
class LinearTrendForecastArtifact:
    """Self-contained JSON-serializable trend forecast artifact (no training data)."""

    artifact_format: str
    schema_version: str
    task_id: str
    model_type: str
    seed: int
    origin_timestamp: str
    train_end_timestamp: str
    n_train_observations: int
    required_fields: tuple[str, ...]
    timestamp_format: str
    intercept: float
    slope: float

    def __post_init__(self) -> None:
        if self.artifact_format != "json":
            raise UnsafeArtifactFormatError(
                f"Only artifact_format='json' is allowed, got {self.artifact_format!r}"
            )
        if self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaVersionMismatchError(
                f"Unsupported schema_version {self.schema_version!r}; "
                f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        if self.model_type != "ols_linear_trend":
            raise ArtifactValidationError(
                f"Unsupported model_type {self.model_type!r}; expected 'ols_linear_trend'"
            )
        if isinstance(self.n_train_observations, bool) or not isinstance(self.n_train_observations, int) or self.n_train_observations < 2:
            raise ArtifactValidationError(
                f"n_train_observations must be int >= 2, got {self.n_train_observations!r}"
            )
        origin = _parse_iso_date(self.origin_timestamp, "origin_timestamp")
        train_end = _parse_iso_date(self.train_end_timestamp, "train_end_timestamp")
        if train_end <= origin:
            raise ArtifactValidationError(
                f"train_end_timestamp ({self.train_end_timestamp}) must be after origin_timestamp ({self.origin_timestamp})"
            )
        if self.required_fields != REQUIRED_INFERENCE_FIELDS:
            raise ArtifactValidationError(
                f"input_schema.required_fields must be {list(REQUIRED_INFERENCE_FIELDS)}, "
                f"got {list(self.required_fields)}"
            )
        if self.timestamp_format != "YYYY-MM-DD":
            raise ArtifactValidationError(
                f"Unsupported timestamp_format {self.timestamp_format!r}; expected 'YYYY-MM-DD'"
            )
        for name, val in (("intercept", self.intercept), ("slope", self.slope)):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                raise ArtifactValidationError(
                    f"Parameter {name!r} must be a finite number, got {val!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_format": self.artifact_format,
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "model_type": self.model_type,
            "seed": int(self.seed),
            "origin_timestamp": self.origin_timestamp,
            "train_end_timestamp": self.train_end_timestamp,
            "n_train_observations": int(self.n_train_observations),
            "input_schema": {
                "required_fields": list(self.required_fields),
                "timestamp_format": self.timestamp_format,
            },
            "parameters": {
                "intercept": float(self.intercept),
                "slope": float(self.slope),
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2) + "\n"

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LinearTrendForecastArtifact":
        if not isinstance(data, Mapping):
            raise ArtifactValidationError(
                f"Artifact payload must be a JSON object, got {type(data).__name__}"
            )
        keys = set(data.keys())
        unknown = sorted(keys - ALLOWED_ARTIFACT_TOP_KEYS)
        if unknown:
            raise ArtifactValidationError(f"Unknown artifact field(s): {unknown}")
        missing = sorted(ALLOWED_ARTIFACT_TOP_KEYS - keys)
        if missing:
            raise ArtifactValidationError(f"Missing required artifact field(s): {missing}")

        if data["artifact_format"] != "json":
            raise UnsafeArtifactFormatError(
                f"Refusing non-JSON artifact_format={data['artifact_format']!r}"
            )
        if data["schema_version"] not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaVersionMismatchError(
                f"Unsupported schema_version {data['schema_version']!r}; "
                f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )

        input_schema = data["input_schema"]
        if not isinstance(input_schema, Mapping) or set(input_schema.keys()) != ALLOWED_INPUT_SCHEMA_KEYS:
            raise ArtifactValidationError(
                f"Invalid input_schema keys: expected {sorted(ALLOWED_INPUT_SCHEMA_KEYS)}"
            )
        req_fields = input_schema["required_fields"]
        if not isinstance(req_fields, list):
            raise ArtifactValidationError("input_schema.required_fields must be a list")

        params = data["parameters"]
        if not isinstance(params, Mapping) or set(params.keys()) != ALLOWED_PARAM_KEYS:
            raise ArtifactValidationError(
                f"Invalid parameters keys: expected {sorted(ALLOWED_PARAM_KEYS)}"
            )

        seed = data["seed"]
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ArtifactValidationError(f"seed must be an int, got {seed!r}")

        return cls(
            artifact_format=str(data["artifact_format"]),
            schema_version=str(data["schema_version"]),
            task_id=str(data["task_id"]),
            model_type=str(data["model_type"]),
            seed=seed,
            origin_timestamp=str(data["origin_timestamp"]),
            train_end_timestamp=str(data["train_end_timestamp"]),
            n_train_observations=data["n_train_observations"],
            required_fields=tuple(req_fields),
            timestamp_format=str(input_schema["timestamp_format"]),
            intercept=params["intercept"],
            slope=params["slope"],
        )


# ── Date & Fixture Validation ────────────────────────────────────────────────


def _parse_iso_date(raw: Any, context: str) -> date:
    if not isinstance(raw, str) or len(raw) != 10:
        raise InferenceSchemaError(
            f"{context}: expected ISO-8601 date string 'YYYY-MM-DD', got {raw!r}"
        )
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise InferenceSchemaError(
            f"{context}: invalid ISO-8601 date {raw!r}"
        ) from exc


def _validate_finite_value(raw: Any, context: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise InferenceSchemaError(
            f"{context}: expected finite number, got {raw!r} ({type(raw).__name__})"
        )
    val = float(raw)
    if not math.isfinite(val):
        raise InferenceSchemaError(f"{context}: value must be finite, got {raw!r}")
    return val


def load_fixture(path: Path | str | None = None) -> dict[str, Any]:
    """Load the ten-day synthetic time-series fixture JSON."""
    fixture_path = Path(path) if path is not None else DEFAULT_FIXTURE_PATH
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not data.get("is_synthetic", False):
        raise ArtifactValidationError("Fixture must explicitly declare is_synthetic=true")
    if data.get("shuffling_allowed", True):
        raise ChronologyError("Time-series fixture must set shuffling_allowed=false")
    return data


def load_schema(path: Path | str | None = None) -> dict[str, Any]:
    """Load the R41-T030 schema JSON."""
    schema_path = Path(path) if path is not None else DEFAULT_SCHEMA_PATH
    return json.loads(schema_path.read_text(encoding="utf-8"))


def split_chronological(
    records: Sequence[Mapping[str, Any]],
    *,
    train_count: int = TRAIN_COUNT,
    val_count: int = VAL_COUNT,
    test_count: int = TEST_COUNT,
) -> TimeSeriesSplits:
    """Validate strictly consecutive daily order and split 6/2/2 without shuffling."""
    expected_total = train_count + val_count + test_count
    if len(records) != expected_total:
        raise ChronologyError(
            f"Expected {expected_total} consecutive daily records, got {len(records)}"
        )

    cleaned: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    prev_date: date | None = None

    for idx, rec in enumerate(records):
        if not isinstance(rec, Mapping):
            raise InferenceSchemaError(f"Row {idx}: record must be a mapping")
        for field in REQUIRED_FIXTURE_FIELDS:
            if field not in rec:
                raise InferenceSchemaError(f"Row {idx}: missing required field {field!r}")
        rec_id = rec["record_id"]
        if not isinstance(rec_id, str) or not rec_id.strip():
            raise InferenceSchemaError(f"Row {idx}: invalid record_id {rec_id!r}")
        if rec_id in seen_ids:
            raise InferenceSchemaError(f"Duplicate record_id {rec_id!r}")
        seen_ids.add(rec_id)

        dt = _parse_iso_date(rec["timestamp"], f"Row {idx} ({rec_id!r})")
        if prev_date is not None and dt != prev_date + timedelta(days=1):
            raise ChronologyError(
                f"Row {idx} ({rec_id!r}): non-consecutive or shuffled timestamp "
                f"{rec['timestamp']!r} after {prev_date.isoformat()!r}"
            )
        prev_date = dt

        val = _validate_finite_value(rec["value"], f"Row {idx} ({rec_id!r})")
        cleaned.append(
            {"record_id": rec_id, "timestamp": rec["timestamp"], "value": val}
        )

    return TimeSeriesSplits(
        train=tuple(cleaned[:train_count]),
        validation=tuple(cleaned[train_count : train_count + val_count]),
        test=tuple(cleaned[train_count + val_count :]),
    )


# ── Fit, Save, Reload & Inference Functions ──────────────────────────────────


def fit_trend_model(
    train_records: Sequence[Mapping[str, Any]],
    *,
    seed: int = FIXED_SEED,
) -> LinearTrendForecastArtifact:
    """Fit closed-form OLS linear trend strictly on chronological training rows."""
    if len(train_records) < 2:
        raise ChronologyError("Need at least 2 chronological training observations to fit trend")

    dates: list[date] = []
    values: list[float] = []
    seen_ids: set[str] = set()
    for idx, rec in enumerate(train_records):
        if not isinstance(rec, Mapping):
            raise InferenceSchemaError(f"Train row {idx}: record must be a mapping")
        for field in REQUIRED_FIXTURE_FIELDS:
            if field not in rec:
                raise InferenceSchemaError(f"Train row {idx}: missing required field {field!r}")
        rec_id = rec["record_id"]
        if not isinstance(rec_id, str) or not rec_id.strip() or rec_id in seen_ids:
            raise InferenceSchemaError(f"Train row {idx}: invalid/duplicate record_id {rec_id!r}")
        seen_ids.add(rec_id)

        dt = _parse_iso_date(rec["timestamp"], f"Train row {idx}")
        if dates and dt != dates[-1] + timedelta(days=1):
            raise ChronologyError(
                f"Train row {idx}: timestamps must be strictly consecutive daily order"
            )
        dates.append(dt)
        values.append(_validate_finite_value(rec["value"], f"Train row {idx}"))

    origin = dates[0]
    t_vals = [float((d - origin).days) for d in dates]
    n = float(len(t_vals))
    t_mean = sum(t_vals) / n
    y_mean = sum(values) / n
    denom = sum((t - t_mean) ** 2 for t in t_vals)
    numer = sum((t - t_mean) * (y - y_mean) for t, y in zip(t_vals, values))
    slope = float(numer / denom)
    intercept = float(y_mean - slope * t_mean)

    return LinearTrendForecastArtifact(
        artifact_format="json",
        schema_version=DEFAULT_SCHEMA_VERSION,
        task_id=TASK_ID,
        model_type="ols_linear_trend",
        seed=seed,
        origin_timestamp=origin.isoformat(),
        train_end_timestamp=dates[-1].isoformat(),
        n_train_observations=len(train_records),
        required_fields=REQUIRED_INFERENCE_FIELDS,
        timestamp_format="YYYY-MM-DD",
        intercept=intercept,
        slope=slope,
    )


def _reject_unsafe_path(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix in FORBIDDEN_EXTENSIONS:
        raise UnsafeArtifactFormatError(
            f"Refusing unsafe pickle/binary artifact format {suffix!r} for {path.name!r}; use '.json'"
        )
    if suffix != ".json":
        raise UnsafeArtifactFormatError(
            f"Artifact path must end with '.json', got {path.name!r}"
        )


def save_artifact(
    artifact: LinearTrendForecastArtifact,
    path: Path | str,
) -> Path:
    """Save the validated forecast artifact as safe JSON (never pickle)."""
    target = Path(path)
    _reject_unsafe_path(target)
    target.write_text(artifact.to_json(), encoding="utf-8")
    return target


def load_artifact(path: Path | str) -> LinearTrendForecastArtifact:
    """Load and validate a JSON forecast artifact (explicitly rejects pickle)."""
    source = Path(path)
    _reject_unsafe_path(source)
    try:
        raw_text = source.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactValidationError(
            f"Artifact {source.name!r} is not valid UTF-8 JSON"
        ) from exc
    return LinearTrendForecastArtifact.from_dict(data)


def predict_with_artifact(
    artifact: LinearTrendForecastArtifact,
    inference_records: Sequence[Mapping[str, Any]],
) -> tuple[ForecastPrediction, ...]:
    """Run schema-validated inference using ONLY the saved artifact (no training data/credentials)."""
    if not inference_records:
        raise InferenceSchemaError("inference_records must not be empty")

    origin = _parse_iso_date(artifact.origin_timestamp, "artifact.origin_timestamp")
    predictions: list[ForecastPrediction] = []
    seen_ids: set[str] = set()
    prev_dt: date | None = None

    for idx, rec in enumerate(inference_records):
        if not isinstance(rec, Mapping):
            raise InferenceSchemaError(
                f"Inference row {idx}: expected mapping, got {type(rec).__name__}"
            )
        for field in artifact.required_fields:
            if field not in rec:
                raise InferenceSchemaError(
                    f"Inference row {idx}: missing required field {field!r}"
                )
        rec_id = rec["record_id"]
        if not isinstance(rec_id, str) or not rec_id.strip():
            raise InferenceSchemaError(f"Inference row {idx}: invalid record_id {rec_id!r}")
        if rec_id in seen_ids:
            raise InferenceSchemaError(f"Duplicate record_id {rec_id!r} in inference batch")
        seen_ids.add(rec_id)

        dt = _parse_iso_date(rec["timestamp"], f"Inference row {idx} ({rec_id!r})")
        if prev_dt is not None and dt <= prev_dt:
            raise ChronologyError(
                f"Inference row {idx} ({rec_id!r}): timestamps must be strictly increasing "
                f"({rec['timestamp']!r} after {prev_dt.isoformat()!r})"
            )
        prev_dt = dt

        day_index = (dt - origin).days
        y_hat = float(artifact.intercept + artifact.slope * float(day_index))
        if not math.isfinite(y_hat):
            raise InferenceSchemaError(f"Non-finite forecast for {rec_id!r}")
        predictions.append(
            ForecastPrediction(
                record_id=rec_id,
                timestamp=rec["timestamp"],
                day_index=day_index,
                predicted_value=y_hat,
            )
        )

    return tuple(predictions)


def run_saved_model_verification(
    fixture_path: Path | str | None = None,
    artifact_path: Path | str | None = None,
) -> dict[str, Any]:
    """Fit on d1..d6, save JSON artifact, reload, and verify before/after predictions."""
    fixture_doc = load_fixture(fixture_path)
    splits = split_chronological(fixture_doc["records"])
    artifact = fit_trend_model(splits.train, seed=int(fixture_doc.get("seed", FIXED_SEED)))

    # Inference-only inputs (strip 'value' so inference never touches target values or train data)
    val_infer_inputs = [{"record_id": r["record_id"], "timestamp": r["timestamp"]} for r in splits.validation]
    test_infer_inputs = [{"record_id": r["record_id"], "timestamp": r["timestamp"]} for r in splits.test]

    before_val = predict_with_artifact(artifact, val_infer_inputs)
    before_test = predict_with_artifact(artifact, test_infer_inputs)

    target_artifact = Path(artifact_path) if artifact_path is not None else DEFAULT_ARTIFACT_PATH
    save_artifact(artifact, target_artifact)
    reloaded = load_artifact(target_artifact)

    after_val = predict_with_artifact(reloaded, val_infer_inputs)
    after_test = predict_with_artifact(reloaded, test_infer_inputs)

    max_reload_diff = 0.0
    for b_p, a_p in zip(before_val + before_test, after_val + after_test):
        max_reload_diff = max(max_reload_diff, abs(b_p.predicted_value - a_p.predicted_value))

    def _mae(preds: Sequence[ForecastPrediction], rows: Sequence[Mapping[str, Any]]) -> float:
        return sum(abs(p.predicted_value - float(r["value"])) for p, r in zip(preds, rows)) / float(len(rows))

    return {
        "task_id": TASK_ID,
        "project_section": PROJECT_SECTION,
        "verification_only": True,
        "synthetic_disclaimer": fixture_doc["disclaimer"],
        "reload_tolerance": RELOAD_TOLERANCE,
        "max_reload_abs_diff": max_reload_diff,
        "reload_within_tolerance": max_reload_diff <= RELOAD_TOLERANCE,
        "artifact": reloaded.to_dict(),
        "splits": {
            "train_ids": [r["record_id"] for r in splits.train],
            "validation": {
                "record_ids": [r["record_id"] for r in splits.validation],
                "actual_values": [r["value"] for r in splits.validation],
                "before_reload": [p.to_dict() for p in before_val],
                "after_reload": [p.to_dict() for p in after_val],
                "mae": _mae(after_val, splits.validation),
            },
            "test": {
                "record_ids": [r["record_id"] for r in splits.test],
                "actual_values": [r["value"] for r in splits.test],
                "before_reload": [p.to_dict() for p in before_test],
                "after_reload": [p.to_dict() for p in after_test],
                "mae": _mae(after_test, splits.test),
            },
        },
    }
