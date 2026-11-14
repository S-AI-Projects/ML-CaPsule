"""Explicit preprocessing pipeline for World happiness report regression (Task R41-T035)."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

FEATURE_ORDER = ("x1", "x2")


@dataclass
class PreprocessedArray:
    """Represents the preprocessed output with explicit shape and dtype assertions."""

    shape: Tuple[int, int]
    dtype: str
    feature_names: List[str]
    values: List[List[float]]
    record_ids: List[str]


class ExplicitPreprocessor:
    """
    Fits imputation and scaling parameters strictly on training records.
    Guarantees that fitting on held-out data is impossible through the transform interface.
    Retains numeric types (float) and strict feature ordering.
    """

    def __init__(self) -> None:
        self.fitted: bool = False
        self.feature_means: Dict[str, float] = {}
        self.feature_stds: Dict[str, float] = {}
        self.train_ids: List[str] = []

    def fit(self, train_records: List[Dict[str, Any]]) -> "ExplicitPreprocessor":
        """Calculates scaling and imputation parameters ONLY on training records."""
        if not train_records:
            raise ValueError("Cannot fit preprocessor on empty training set")

        self.train_ids = [r["record_id"] for r in train_records]

        for feature in FEATURE_ORDER:
            vals = []
            for r in train_records:
                if feature not in r or r[feature] is None:
                    continue  # missing values handled during transform by imputation
                val = float(r[feature])
                if not math.isfinite(val):
                    raise ValueError(f"Contaminated input: non-finite value in training feature {feature}")
                vals.append(val)

            if not vals:
                raise ValueError(f"Feature {feature} has no valid values in training set")

            mean_val = sum(vals) / len(vals)
            variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
            std_val = math.sqrt(variance)

            self.feature_means[feature] = mean_val
            # If std is 0 (constant feature), use 1.0 to avoid division by zero
            self.feature_stds[feature] = std_val if std_val > 1e-12 else 1.0

        self.fitted = True
        return self

    def transform(self, records: List[Dict[str, Any]]) -> PreprocessedArray:
        """
        Transforms input records using previously fitted parameters.
        Does NOT update or refit parameters on incoming data.
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fit on training rows before transform() can be called")

        if not records:
            return PreprocessedArray(
                shape=(0, len(FEATURE_ORDER)),
                dtype="float64",
                feature_names=list(FEATURE_ORDER),
                values=[],
                record_ids=[],
            )

        transformed_rows: List[List[float]] = []
        ids: List[str] = []

        for r in records:
            ids.append(r.get("record_id", "unknown"))
            row_vals: List[float] = []

            for feature in FEATURE_ORDER:
                if feature not in r or r[feature] is None:
                    # Impute with training mean
                    raw_val = self.feature_means[feature]
                else:
                    try:
                        raw_val = float(r[feature])
                    except (ValueError, TypeError):
                        raise ValueError(f"Contamination fixture: non-numeric value in feature '{feature}'")

                if not math.isfinite(raw_val):
                    raise ValueError(f"Contamination fixture: non-finite value in feature '{feature}'")

                # Scale using training stats
                scaled = (raw_val - self.feature_means[feature]) / self.feature_stds[feature]
                row_vals.append(float(scaled))

            transformed_rows.append(row_vals)

        return PreprocessedArray(
            shape=(len(transformed_rows), len(FEATURE_ORDER)),
            dtype="float64",
            feature_names=list(FEATURE_ORDER),
            values=transformed_rows,
            record_ids=ids,
        )

    def fit_transform(self, train_records: List[Dict[str, Any]]) -> PreprocessedArray:
        """Convenience method for training partition."""
        return self.fit(train_records).transform(train_records)


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load JSON fixture."""
    if path is None:
        path = Path(__file__).parent / "fixture.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
