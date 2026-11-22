"""Reproducible configuration & bounded SVM pipeline for R41-T027.

Task: R41-T027 · 7 of 10
Project/section: Banknote authentication – SVM
Student: Sharon Robinson
Repository: S-AI-Projects/ML-CaPsule

SYNTHETIC DATA NOTICE
=====================
All default fixture records and verification outputs produced by this module
use an eight-row synthetic classification fixture (x1=[0..7], x2=1,
label=[0,0,0,0,1,1,1,1]). They are NOT observations from the UCI Banknote
Authentication dataset and must NOT be cited as evidence of real-world or
clinical performance.

Design Guarantees
-----------------
1. Serializable configuration (``ExperimentConfig``): captures ``schema_version``,
   ``task_id``, ``seed``, ``feature_ordering``, ``valid_labels``, ``split_ids``
   (train/validation/test), and ``algorithm`` settings; supports lossless JSON
   save/load round-trips.
2. Strict schema validation:
   - Rejects any unknown field at top level, inside ``split_ids``, or inside
     ``algorithm`` (``UnknownConfigFieldError``).
   - Rejects unsupported ``schema_version`` (``SchemaVersionError``).
   - Rejects contradictory settings such as ``kernel="linear"`` with non-null
     ``gamma``, ``kernel="rbf"`` without positive ``gamma``, ``name="linear_svm"``
     with a non-linear kernel, overlapping ``split_ids``, or invalid
     hyperparameters (``ContradictoryConfigError``).
3. Shape and split enforcement:
   - Reconstructs exact disjoint, group-isolated, stratified train/val/test
     partitions strictly from ``config.split_ids``.
   - Rejects incompatible input shapes (wrong column count, missing/extra
     features, or swapped feature order) with ``IncompatibleInputShapeError``.
4. Reproducibility & provenance:
   - Reloading a serialized config reconstructs the exact same splits, learned
     weights, decision scores, and predictions on CPU with Python 3 stdlib only.
   - Records exact Python/platform environment versions in every run report.
"""

from __future__ import annotations

import json
import math
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# ── Constants ────────────────────────────────────────────────────────────────

TASK_ID: str = "R41-T027"
PROJECT_SECTION: str = "Banknote authentication – SVM"
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset({"1.0.0"})
DEFAULT_SCHEMA_VERSION: str = "1.0.0"
FIXED_SEED: int = 42
EXPECTED_FEATURE_ORDERING: tuple[str, ...] = ("x1", "x2")
EXPECTED_VALID_LABELS: tuple[int, ...] = (0, 1)
ALLOWED_TOP_LEVEL_FIELDS: frozenset[str] = frozenset(
    {
        "schema_version",
        "task_id",
        "seed",
        "feature_ordering",
        "valid_labels",
        "split_ids",
        "algorithm",
    }
)
ALLOWED_SPLIT_FIELDS: frozenset[str] = frozenset({"train", "validation", "test"})
ALLOWED_ALGORITHM_FIELDS: frozenset[str] = frozenset(
    {"name", "kernel", "c", "learning_rate", "max_iter", "gamma"}
)
ALLOWED_ALGORITHM_NAMES: frozenset[str] = frozenset({"linear_svm", "kernel_svm"})
ALLOWED_KERNELS: frozenset[str] = frozenset({"linear", "rbf"})
METADATA_FIELDS: frozenset[str] = frozenset({"record_id", "group_id", "label"})
MAX_ITER_LIMIT: int = 10_000

LAB_DIR: Path = Path(__file__).resolve().parent
DEFAULT_FIXTURE_PATH: Path = LAB_DIR / "fixture.json"
DEFAULT_CONFIG_PATH: Path = LAB_DIR / "config.json"
DEFAULT_SCHEMA_PATH: Path = LAB_DIR / "schema.json"


# ── Exceptions ───────────────────────────────────────────────────────────────


class ConfigReproducibilityError(ValueError):
    """Base exception for R41-T027 configuration and pipeline errors."""


class UnknownConfigFieldError(ConfigReproducibilityError):
    """Raised when a configuration object contains unrecognized fields."""


class SchemaVersionError(ConfigReproducibilityError):
    """Raised when schema_version is missing or unsupported."""


class ContradictoryConfigError(ConfigReproducibilityError):
    """Raised when configuration fields are internally contradictory or invalid."""


class IncompatibleInputShapeError(ConfigReproducibilityError):
    """Raised when input records or feature matrices violate expected feature shape."""


class SplitIntegrityError(ConfigReproducibilityError):
    """Raised when split IDs violate disjointness, coverage, stratification, or group isolation."""


class ModelNotFittedError(RuntimeError):
    """Raised when predict or decision_function is called before fitting."""


# ── Validation Helpers ───────────────────────────────────────────────────────


def _check_exact_fields(
    data: Mapping[str, Any],
    allowed: frozenset[str],
    context: str,
) -> None:
    if not isinstance(data, Mapping):
        raise ContradictoryConfigError(
            f"{context} must be a JSON object/mapping, got {type(data).__name__}"
        )
    keys = set(data.keys())
    unknown = sorted(keys - allowed)
    if unknown:
        raise UnknownConfigFieldError(
            f"Unknown field(s) in {context}: {unknown}. Allowed: {sorted(allowed)}"
        )
    missing = sorted(allowed - keys)
    if missing:
        raise ContradictoryConfigError(
            f"Missing required field(s) in {context}: {missing}"
        )


def _validate_int(
    value: Any,
    field_name: str,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContradictoryConfigError(
            f"{field_name!r} must be an integer, got {value!r} ({type(value).__name__})"
        )
    if min_value is not None and value < min_value:
        raise ContradictoryConfigError(
            f"{field_name!r} must be >= {min_value}, got {value}"
        )
    if max_value is not None and value > max_value:
        raise ContradictoryConfigError(
            f"{field_name!r} must be <= {max_value}, got {value}"
        )
    return value


def _validate_positive_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContradictoryConfigError(
            f"{field_name!r} must be a positive finite number, got {value!r}"
        )
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ContradictoryConfigError(
            f"{field_name!r} must be > 0 and finite, got {value!r}"
        )
    return numeric


def _validate_finite_feature(value: Any, field_name: str, row_idx: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise IncompatibleInputShapeError(
            f"Row {row_idx}: feature {field_name!r} must be numeric, "
            f"got {value!r} ({type(value).__name__})"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise IncompatibleInputShapeError(
            f"Row {row_idx}: feature {field_name!r} must be finite, got {value!r}"
        )
    return numeric


# ── Serializable Configuration Objects ───────────────────────────────────────


@dataclass(frozen=True)
class SVMConfig:
    """Validated, serializable SVM algorithm settings."""

    name: str
    kernel: str
    c: float
    learning_rate: float
    max_iter: int
    gamma: float | None = None

    def __post_init__(self) -> None:
        if self.name not in ALLOWED_ALGORITHM_NAMES:
            raise ContradictoryConfigError(
                f"Unsupported algorithm name {self.name!r}; allowed: {sorted(ALLOWED_ALGORITHM_NAMES)}"
            )
        if self.kernel not in ALLOWED_KERNELS:
            raise ContradictoryConfigError(
                f"Unsupported kernel {self.kernel!r}; allowed: {sorted(ALLOWED_KERNELS)}"
            )
        if self.name == "linear_svm" and self.kernel != "linear":
            raise ContradictoryConfigError(
                f"Contradictory algorithm settings: name='linear_svm' requires kernel='linear', "
                f"got kernel={self.kernel!r}"
            )
        _validate_positive_float(self.c, "algorithm.c")
        _validate_positive_float(self.learning_rate, "algorithm.learning_rate")
        _validate_int(self.max_iter, "algorithm.max_iter", min_value=1, max_value=MAX_ITER_LIMIT)

        if self.kernel == "linear" and self.gamma is not None:
            raise ContradictoryConfigError(
                f"Contradictory algorithm settings: kernel='linear' must set gamma=None, "
                f"got gamma={self.gamma!r}"
            )
        if self.kernel == "rbf":
            if self.gamma is None:
                raise ContradictoryConfigError(
                    "Contradictory algorithm settings: kernel='rbf' requires positive float gamma"
                )
            _validate_positive_float(self.gamma, "algorithm.gamma")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kernel": self.kernel,
            "c": float(self.c),
            "learning_rate": float(self.learning_rate),
            "max_iter": int(self.max_iter),
            "gamma": None if self.gamma is None else float(self.gamma),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SVMConfig":
        _check_exact_fields(data, ALLOWED_ALGORITHM_FIELDS, "algorithm")
        gamma_raw = data["gamma"]
        gamma_val = None if gamma_raw is None else _validate_positive_float(gamma_raw, "algorithm.gamma")
        return cls(
            name=str(data["name"]),
            kernel=str(data["kernel"]),
            c=_validate_positive_float(data["c"], "algorithm.c"),
            learning_rate=_validate_positive_float(data["learning_rate"], "algorithm.learning_rate"),
            max_iter=_validate_int(
                data["max_iter"], "algorithm.max_iter", min_value=1, max_value=MAX_ITER_LIMIT
            ),
            gamma=gamma_val,
        )


@dataclass(frozen=True)
class SplitIDsConfig:
    """Validated, pairwise-disjoint train/validation/test record ID lists."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        buckets = {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }
        seen_all: dict[str, str] = {}
        for split_name, ids in buckets.items():
            if not ids:
                raise ContradictoryConfigError(
                    f"split_ids.{split_name} must not be empty"
                )
            seen_split: set[str] = set()
            for rec_id in ids:
                if not isinstance(rec_id, str) or not rec_id.strip():
                    raise ContradictoryConfigError(
                        f"split_ids.{split_name} contains invalid record_id {rec_id!r}"
                    )
                if rec_id in seen_split:
                    raise ContradictoryConfigError(
                        f"Duplicate record_id {rec_id!r} in split_ids.{split_name}"
                    )
                if rec_id in seen_all:
                    raise ContradictoryConfigError(
                        f"Contradictory split_ids: record_id {rec_id!r} appears in both "
                        f"{seen_all[rec_id]!r} and {split_name!r}"
                    )
                seen_split.add(rec_id)
                seen_all[rec_id] = split_name

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "train": list(self.train),
            "validation": list(self.validation),
            "test": list(self.test),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SplitIDsConfig":
        _check_exact_fields(data, ALLOWED_SPLIT_FIELDS, "split_ids")
        parsed: dict[str, tuple[str, ...]] = {}
        for split_name in ("train", "validation", "test"):
            raw = data[split_name]
            if not isinstance(raw, list):
                raise ContradictoryConfigError(
                    f"split_ids.{split_name} must be a list of strings, got {type(raw).__name__}"
                )
            parsed[split_name] = tuple(raw)
        return cls(
            train=parsed["train"],
            validation=parsed["validation"],
            test=parsed["test"],
        )


@dataclass(frozen=True)
class ExperimentConfig:
    """Strict, serializable experiment configuration for R41-T027."""

    schema_version: str
    task_id: str
    seed: int
    feature_ordering: tuple[str, ...]
    valid_labels: tuple[int, ...]
    split_ids: SplitIDsConfig
    algorithm: SVMConfig

    def __post_init__(self) -> None:
        if not isinstance(self.schema_version, str) or self.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaVersionError(
                f"Unsupported schema_version {self.schema_version!r}; "
                f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ContradictoryConfigError(f"task_id must be a non-empty string, got {self.task_id!r}")
        _validate_int(self.seed, "seed", min_value=0)

        if not self.feature_ordering:
            raise ContradictoryConfigError("feature_ordering must not be empty")
        if any(not isinstance(f, str) or not f.strip() for f in self.feature_ordering):
            raise ContradictoryConfigError(
                f"feature_ordering must contain non-empty strings, got {self.feature_ordering!r}"
            )
        if len(set(self.feature_ordering)) != len(self.feature_ordering):
            raise ContradictoryConfigError(
                f"feature_ordering contains duplicate feature names: {self.feature_ordering!r}"
            )

        if len(self.valid_labels) != 2 or tuple(sorted(set(self.valid_labels))) != EXPECTED_VALID_LABELS:
            raise ContradictoryConfigError(
                f"Binary SVM requires valid_labels={list(EXPECTED_VALID_LABELS)}, got {list(self.valid_labels)!r}"
            )
        if any(isinstance(lbl, bool) or not isinstance(lbl, int) for lbl in self.valid_labels):
            raise ContradictoryConfigError("valid_labels must be non-bool integers")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "seed": int(self.seed),
            "feature_ordering": list(self.feature_ordering),
            "valid_labels": list(self.valid_labels),
            "split_ids": self.split_ids.to_dict(),
            "algorithm": self.algorithm.to_dict(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n"

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentConfig":
        _check_exact_fields(data, ALLOWED_TOP_LEVEL_FIELDS, "config")
        schema_version = data["schema_version"]
        if not isinstance(schema_version, str) or schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            raise SchemaVersionError(
                f"Unsupported schema_version {schema_version!r}; "
                f"supported: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        feat_raw = data["feature_ordering"]
        if not isinstance(feat_raw, list):
            raise ContradictoryConfigError("feature_ordering must be a list of strings")
        labels_raw = data["valid_labels"]
        if not isinstance(labels_raw, list) or any(
            isinstance(lbl, bool) or not isinstance(lbl, int) for lbl in labels_raw
        ):
            raise ContradictoryConfigError("valid_labels must be a list of integers")
        return cls(
            schema_version=schema_version,
            task_id=data["task_id"],
            seed=_validate_int(data["seed"], "seed", min_value=0),
            feature_ordering=tuple(feat_raw),
            valid_labels=tuple(labels_raw),
            split_ids=SplitIDsConfig.from_dict(data["split_ids"]),
            algorithm=SVMConfig.from_dict(data["algorithm"]),
        )

    @classmethod
    def from_json(cls, payload: str) -> "ExperimentConfig":
        parsed = json.loads(payload)
        return cls.from_dict(parsed)

    @classmethod
    def load(cls, path: Path | str | None = None) -> "ExperimentConfig":
        cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        return cls.from_json(cfg_path.read_text(encoding="utf-8"))


# ── Split Reconstruction & Feature Extraction ────────────────────────────────


@dataclass(frozen=True)
class ReconstructedSplits:
    """Explicit train/validation/test partitions reconstructed from config.split_ids."""

    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    test: tuple[dict[str, Any], ...]


def load_fixture(path: Path | str | None = None) -> dict[str, Any]:
    """Load the eight-row synthetic classification fixture from disk."""
    fixture_path = Path(path) if path is not None else DEFAULT_FIXTURE_PATH
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not data.get("is_synthetic", False):
        raise ContradictoryConfigError("Fixture must explicitly declare is_synthetic=true")
    return data


def load_schema(path: Path | str | None = None) -> dict[str, Any]:
    """Load the R41-T027 schema JSON from disk."""
    schema_path = Path(path) if path is not None else DEFAULT_SCHEMA_PATH
    return json.loads(schema_path.read_text(encoding="utf-8"))


def extract_feature_matrix(
    inputs: Sequence[Mapping[str, Any] | Sequence[float]],
    feature_ordering: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    """Extract a validated 2-D float64 feature matrix matching feature_ordering.

    Accepts either:
      - A sequence of record mappings whose non-metadata keys must match
        ``feature_ordering`` in exact order (no missing, extra, or swapped keys).
      - A sequence of numeric rows whose width must equal ``len(feature_ordering)``.
    """
    if not inputs:
        raise IncompatibleInputShapeError("Input batch must not be empty")

    expected_width = len(feature_ordering)
    matrix: list[tuple[float, ...]] = []

    for row_idx, item in enumerate(inputs):
        if isinstance(item, Mapping):
            actual_feature_keys = tuple(k for k in item.keys() if k not in METADATA_FIELDS)
            if actual_feature_keys != feature_ordering:
                raise IncompatibleInputShapeError(
                    f"Row {row_idx}: incompatible feature keys/order {actual_feature_keys}; "
                    f"expected {feature_ordering}"
                )
            row_vals = tuple(
                _validate_finite_feature(item[fname], fname, row_idx)
                for fname in feature_ordering
            )
            matrix.append(row_vals)
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
            if len(item) != expected_width:
                raise IncompatibleInputShapeError(
                    f"Row {row_idx}: incompatible input width {len(item)}; "
                    f"expected {expected_width} for features {feature_ordering}"
                )
            row_vals = tuple(
                _validate_finite_feature(val, feature_ordering[col_idx], row_idx)
                for col_idx, val in enumerate(item)
            )
            matrix.append(row_vals)
        else:
            raise IncompatibleInputShapeError(
                f"Row {row_idx}: unsupported input row type {type(item).__name__}"
            )

    return tuple(matrix)


def reconstruct_splits(
    records: Sequence[Mapping[str, Any]],
    config: ExperimentConfig,
) -> ReconstructedSplits:
    """Reconstruct exact train/validation/test splits from config.split_ids."""
    if not records:
        raise SplitIntegrityError("Fixture records must not be empty")

    # Validate feature compatibility on all records first
    extract_feature_matrix(records, config.feature_ordering)

    allowed_labels = set(config.valid_labels)
    by_id: dict[str, dict[str, Any]] = {}

    for row_idx, rec in enumerate(records):
        rec_id = rec.get("record_id")
        grp_id = rec.get("group_id")
        if not isinstance(rec_id, str) or not rec_id.strip():
            raise SplitIntegrityError(f"Row {row_idx}: invalid record_id {rec_id!r}")
        if not isinstance(grp_id, str) or not grp_id.strip():
            raise SplitIntegrityError(f"Row {row_idx}: invalid group_id {grp_id!r}")
        if rec_id in by_id:
            raise SplitIntegrityError(f"Duplicate record_id {rec_id!r} in fixture records")

        label = rec.get("label")
        if isinstance(label, bool) or not isinstance(label, int) or label not in allowed_labels:
            raise SplitIntegrityError(
                f"Row {row_idx} ({rec_id!r}): invalid label {label!r}; expected {sorted(allowed_labels)}"
            )
        by_id[rec_id] = dict(rec)

    configured_ids = (
        set(config.split_ids.train)
        | set(config.split_ids.validation)
        | set(config.split_ids.test)
    )
    missing_in_fixture = sorted(configured_ids - set(by_id.keys()))
    if missing_in_fixture:
        raise SplitIntegrityError(
            f"Config split_ids reference unknown record_id(s): {missing_in_fixture}"
        )
    unassigned = sorted(set(by_id.keys()) - configured_ids)
    if unassigned:
        raise SplitIntegrityError(
            f"Fixture record_id(s) not assigned in config split_ids: {unassigned}"
        )

    train_rows = tuple(by_id[rid] for rid in config.split_ids.train)
    val_rows = tuple(by_id[rid] for rid in config.split_ids.validation)
    test_rows = tuple(by_id[rid] for rid in config.split_ids.test)

    # Verify group isolation and class stratification across reconstructed splits
    group_owner: dict[str, str] = {}
    for split_name, split_rows in (
        ("train", train_rows),
        ("validation", val_rows),
        ("test", test_rows),
    ):
        observed_labels = {r["label"] for r in split_rows}
        if observed_labels != allowed_labels:
            raise SplitIntegrityError(
                f"Reconstructed split {split_name!r} is not stratified over {sorted(allowed_labels)}: "
                f"observed {sorted(observed_labels)}"
            )
        for row in split_rows:
            gid = row["group_id"]
            if gid in group_owner and group_owner[gid] != split_name:
                raise SplitIntegrityError(
                    f"Group leakage detected: group_id {gid!r} appears in both "
                    f"{group_owner[gid]!r} and {split_name!r}"
                )
            group_owner[gid] = split_name

    return ReconstructedSplits(train=train_rows, validation=val_rows, test=test_rows)


# ── Bounded Deterministic Linear SVM Classifier ──────────────────────────────


class LinearSVMClassifier:
    """Deterministic L2-regularized hinge-loss linear SVM (CPU, Python stdlib).

    Minimizes ``0.5 * ||w||^2 + C * mean(max(0, 1 - y_pm * (w·x + b)))`` via
    fixed-seed subgradient descent strictly on training rows.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        if config.algorithm.kernel != "linear":
            raise ContradictoryConfigError(
                f"LinearSVMClassifier supports kernel='linear' only, got {config.algorithm.kernel!r}"
            )
        self.config: ExperimentConfig = config
        self.weights_: tuple[float, ...] | None = None
        self.bias_: float | None = None
        self._is_fitted: bool = False

    def fit(self, train_records: Sequence[Mapping[str, Any]]) -> "LinearSVMClassifier":
        """Fit linear SVM weights strictly on training rows."""
        X = extract_feature_matrix(train_records, self.config.feature_ordering)
        allowed = set(self.config.valid_labels)
        y_pm: list[float] = []
        for row_idx, rec in enumerate(train_records):
            lbl = rec.get("label")
            if isinstance(lbl, bool) or not isinstance(lbl, int) or lbl not in allowed:
                raise SplitIntegrityError(
                    f"Row {row_idx}: invalid training label {lbl!r}; expected {sorted(allowed)}"
                )
            y_pm.append(1.0 if lbl == 1 else -1.0)

        n_samples = len(X)
        n_features = len(self.config.feature_ordering)
        w = [0.0] * n_features
        b = 0.0
        rng = random.Random(self.config.seed)
        c_val = self.config.algorithm.c
        lr = self.config.algorithm.learning_rate

        for _ in range(self.config.algorithm.max_iter):
            order = list(range(n_samples))
            rng.shuffle(order)
            grad_w = [w_j for w_j in w]  # L2 gradient of 0.5 * ||w||^2
            grad_b = 0.0
            scale = c_val / float(n_samples)
            for idx in order:
                x_row = X[idx]
                y_i = y_pm[idx]
                margin = sum(w[j] * x_row[j] for j in range(n_features)) + b
                if y_i * margin < 1.0:
                    for j in range(n_features):
                        grad_w[j] -= scale * y_i * x_row[j]
                    grad_b -= scale * y_i
            for j in range(n_features):
                w[j] = float(w[j] - lr * grad_w[j])
            b = float(b - lr * grad_b)

        self.weights_ = tuple(w)
        self.bias_ = b
        self._is_fitted = True
        return self

    def decision_function(
        self,
        inputs: Sequence[Mapping[str, Any] | Sequence[float]],
    ) -> tuple[float, ...]:
        """Compute finite signed decision margins after shape validation."""
        if not self._is_fitted or self.weights_ is None or self.bias_ is None:
            raise ModelNotFittedError("Call .fit() before .decision_function()")
        X = extract_feature_matrix(inputs, self.config.feature_ordering)
        scores = tuple(
            float(sum(w_j * x_j for w_j, x_j in zip(self.weights_, row)) + self.bias_)
            for row in X
        )
        return scores

    def predict(
        self,
        inputs: Sequence[Mapping[str, Any] | Sequence[float]],
    ) -> tuple[int, ...]:
        """Predict binary labels {0, 1} from validated inputs."""
        scores = self.decision_function(inputs)
        return tuple(1 if s >= 0.0 else 0 for s in scores)


# ── Environment Provenance & End-to-End Runner ───────────────────────────────


def collect_environment_versions() -> dict[str, Any]:
    """Capture exact runtime environment versions for reproducibility."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_full": sys.version.splitlines()[0],
        "platform": platform.platform(),
        "stdlib_only": True,
        "schema_version": DEFAULT_SCHEMA_VERSION,
    }


def run_configured_experiment(
    config: ExperimentConfig,
    fixture_doc: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the full reproducible SVM pipeline strictly from a validated config."""
    doc = fixture_doc if fixture_doc is not None else load_fixture()
    splits = reconstruct_splits(doc["records"], config)
    clf = LinearSVMClassifier(config).fit(splits.train)

    split_outputs: dict[str, Any] = {}
    for split_name, split_rows in (
        ("train", splits.train),
        ("validation", splits.validation),
        ("test", splits.test),
    ):
        scores = clf.decision_function(split_rows)
        preds = clf.predict(split_rows)
        labels = [r["label"] for r in split_rows]
        accuracy = sum(int(p == y) for p, y in zip(preds, labels)) / float(len(labels))
        split_outputs[split_name] = {
            "record_ids": [r["record_id"] for r in split_rows],
            "group_ids": [r["group_id"] for r in split_rows],
            "labels": labels,
            "decision_scores": list(scores),
            "predictions": list(preds),
            "prediction_shape": [len(preds)],
            "accuracy": accuracy,
        }

    return {
        "task_id": config.task_id,
        "project_section": PROJECT_SECTION,
        "verification_only": True,
        "synthetic_disclaimer": doc["disclaimer"],
        "config": config.to_dict(),
        "environment": collect_environment_versions(),
        "model_parameters": {
            "weights": list(clf.weights_),
            "bias": float(clf.bias_),
        },
        "splits": split_outputs,
    }
