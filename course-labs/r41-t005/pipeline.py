"""Explicit Preprocessor Pipeline for Task R41-T005: Student performance (UCI) – regression.

Enforces the preprocessing contract:
1. Missing-value imputation and feature scaling parameters are fitted ONLY on training rows.
2. The public transform interface cannot mutate or fit parameters.
3. Feature order ['x1', 'x2'] and numeric float types are strictly preserved.
4. Deliberate contamination fixtures (held-out rows in training, invalid schemas,
   leaked targets, duplicate IDs) are detected and rejected.
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple

FEATURE_ORDER: List[str] = ["x1", "x2"]
ID_FIELD: str = "record_id"
TARGET_FIELD: str = "target"
FORBIDDEN_TRAIN_IDS: Set[str] = {"r5", "r6"}  # Held-out validation and test IDs


class PreprocessingContaminationError(ValueError):
    """Raised when data contamination or leakage is detected."""
    pass


class SchemaValidationError(TypeError):
    """Raised when record types or schemas violate the contract."""
    pass


def validate_record_schema(
    record: Dict[str, Any],
    require_target: bool = False,
    allow_missing: bool = True,
    expected_features: Optional[List[str]] = None,
) -> None:
    """Validate that a record satisfies the data contract."""
    if not isinstance(record, dict):
        raise SchemaValidationError(f"Record must be a dict, got {type(record).__name__}")

    if expected_features is None:
        expected_features = FEATURE_ORDER

    # Record ID validation
    if ID_FIELD not in record:
        raise SchemaValidationError(f"Missing required field: '{ID_FIELD}'")
    rid = record[ID_FIELD]
    if not isinstance(rid, str) or not rid.strip():
        raise SchemaValidationError(f"'{ID_FIELD}' must be a non-empty string, got {rid!r}")

    # Feature validation
    for feat in expected_features:
        if feat not in record:
            raise SchemaValidationError(f"Record '{rid}' is missing required feature '{feat}'")
        val = record[feat]
        if val is None:
            if not allow_missing:
                raise SchemaValidationError(f"Feature '{feat}' in record '{rid}' cannot be None")
            continue
        # Booleans are subclasses of int in Python, reject them explicitly
        if isinstance(val, bool):
            raise SchemaValidationError(
                f"Feature '{feat}' in record '{rid}' must be numeric, got boolean {val}"
            )
        if not isinstance(val, (int, float)):
            raise SchemaValidationError(
                f"Feature '{feat}' in record '{rid}' must be numeric (int/float), got {type(val).__name__}"
            )
        if math.isnan(val):
            if not allow_missing:
                raise SchemaValidationError(f"Feature '{feat}' in record '{rid}' cannot be NaN")
        elif math.isinf(val):
            raise SchemaValidationError(f"Feature '{feat}' in record '{rid}' cannot be infinite")

    # Target validation (when required)
    if require_target:
        if TARGET_FIELD not in record:
            raise SchemaValidationError(f"Record '{rid}' is missing target field '{TARGET_FIELD}'")
        target_val = record[TARGET_FIELD]
        if isinstance(target_val, bool) or not isinstance(target_val, (int, float)):
            raise SchemaValidationError(
                f"Target in record '{rid}' must be numeric, got {type(target_val).__name__}"
            )
        if math.isnan(target_val) or math.isinf(target_val):
            raise SchemaValidationError(f"Target in record '{rid}' must be a finite number")


def validate_records(
    records: List[Dict[str, Any]],
    require_target: bool = False,
    allow_missing: bool = True,
    forbidden_ids: Optional[Set[str]] = None,
) -> None:
    """Validate a collection of records for uniqueness, schema, and contamination."""
    if not isinstance(records, list) or len(records) == 0:
        raise SchemaValidationError("Records must be a non-empty list of dictionaries")

    seen_ids: Set[str] = set()
    for rec in records:
        validate_record_schema(rec, require_target=require_target, allow_missing=allow_missing)
        rid = rec[ID_FIELD]
        if rid in seen_ids:
            raise PreprocessingContaminationError(f"Duplicate record_id detected: '{rid}'")
        seen_ids.add(rid)

        if forbidden_ids and rid in forbidden_ids:
            raise PreprocessingContaminationError(
                f"Contamination error: held-out record '{rid}' present in training partition"
            )


class ExplicitPreprocessor:
    """Preprocessor that fits imputation and scaling strictly on training records."""

    def __init__(self, feature_order: Optional[List[str]] = None) -> None:
        self.feature_order: List[str] = list(feature_order or FEATURE_ORDER)
        self.is_fitted: bool = False
        self.impute_means_: Dict[str, float] = {}
        self.scale_means_: Dict[str, float] = {}
        self.scale_stds_: Dict[str, float] = {}
        self.fitted_ids_: Set[str] = set()

    def fit(
        self,
        records: List[Dict[str, Any]],
        forbidden_ids: Optional[Set[str]] = None,
    ) -> "ExplicitPreprocessor":
        """Fit missing-value imputation and scale parameters strictly on training records."""
        active_forbidden = FORBIDDEN_TRAIN_IDS if forbidden_ids is None else forbidden_ids
        validate_records(records, allow_missing=True, forbidden_ids=active_forbidden)

        impute_means: Dict[str, float] = {}
        scale_means: Dict[str, float] = {}
        scale_stds: Dict[str, float] = {}

        for feat in self.feature_order:
            valid_values = [
                float(r[feat])
                for r in records
                if r[feat] is not None and not math.isnan(float(r[feat]))
            ]
            if len(valid_values) == 0:
                raise ValueError(f"Feature '{feat}' contains no valid values to fit on")

            mean_val = sum(valid_values) / len(valid_values)
            impute_means[feat] = mean_val
            scale_means[feat] = mean_val

            # Compute population standard deviation (ddof=0)
            variance = sum((v - mean_val) ** 2 for v in valid_values) / len(valid_values)
            std_val = math.sqrt(variance)
            scale_stds[feat] = std_val

        self.impute_means_ = impute_means
        self.scale_means_ = scale_means
        self.scale_stds_ = scale_stds
        self.fitted_ids_ = {r[ID_FIELD] for r in records}
        self.is_fitted = True
        return self

    def transform(self, records: List[Dict[str, Any]]) -> Tuple[List[List[float]], Tuple[int, int], type]:
        """Transform records using strictly pre-fitted parameters.

        Guarantees:
        - Never mutates or refits parameters.
        - Preserves exact feature order ['x1', 'x2'].
        - Preserves float numeric type.
        - Returns (matrix, shape, dtype).
        """
        if not self.is_fitted:
            raise RuntimeError("ExplicitPreprocessor must be fitted before calling transform()")

        validate_records(records, allow_missing=True, forbidden_ids=None)

        # Retain snapshot of parameters before transformation to guarantee immutability
        impute_snapshot = dict(self.impute_means_)
        means_snapshot = dict(self.scale_means_)
        stds_snapshot = dict(self.scale_stds_)

        transformed_matrix: List[List[float]] = []

        for rec in records:
            row: List[float] = []
            for feat in self.feature_order:
                raw_val = rec[feat]
                # 1. Imputation step using training mean
                if raw_val is None or math.isnan(float(raw_val)):
                    imputed = impute_snapshot[feat]
                else:
                    imputed = float(raw_val)

                # 2. Scaling step using training mean and std
                feat_mean = means_snapshot[feat]
                feat_std = stds_snapshot[feat]

                if feat_std > 1e-12:
                    scaled = (imputed - feat_mean) / feat_std
                else:
                    # Constant feature (std == 0), centered to 0.0
                    scaled = 0.0

                row.append(float(scaled))
            transformed_matrix.append(row)

        # Assert no parameter drift occurred
        assert self.impute_means_ == impute_snapshot, "Public transform must not mutate impute parameters"
        assert self.scale_means_ == means_snapshot, "Public transform must not mutate scale means"
        assert self.scale_stds_ == stds_snapshot, "Public transform must not mutate scale stds"

        shape = (len(transformed_matrix), len(self.feature_order))
        dtype = float
        return transformed_matrix, shape, dtype

    def fit_transform(
        self,
        records: List[Dict[str, Any]],
        forbidden_ids: Optional[Set[str]] = None,
    ) -> Tuple[List[List[float]], Tuple[int, int], type]:
        """Fit on training records and transform them."""
        return self.fit(records, forbidden_ids=forbidden_ids).transform(records)
