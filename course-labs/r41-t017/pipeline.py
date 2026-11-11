"""Reproducible PCA configuration and eigenfaces projection pipeline (Task R41-T017)."""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ALLOWED_CONFIG_KEYS = {"schema_version", "seed", "feature_ordering", "split_ids", "algorithm_settings"}
ALLOWED_ALGO_KEYS = {"n_components", "tolerance", "unit_vector"}


class ReproduciblePCAConfig:
    """Validates and enforces strict serializable configuration for PCA projections."""

    def __init__(self, raw_config: Dict[str, Any]) -> None:
        self.validate(raw_config)
        self.schema_version = raw_config["schema_version"]
        self.seed = raw_config["seed"]
        self.feature_ordering = list(raw_config["feature_ordering"])
        self.split_ids = raw_config["split_ids"]
        self.algorithm_settings = raw_config["algorithm_settings"]

    @staticmethod
    def validate(raw_config: Dict[str, Any]) -> None:
        extra_keys = set(raw_config.keys()) - ALLOWED_CONFIG_KEYS
        if extra_keys:
            raise ValueError(f"Unknown configuration fields detected: {extra_keys}")

        missing_keys = ALLOWED_CONFIG_KEYS - set(raw_config.keys())
        if missing_keys:
            raise ValueError(f"Missing required configuration fields: {missing_keys}")

        algo = raw_config["algorithm_settings"]
        extra_algo_keys = set(algo.keys()) - ALLOWED_ALGO_KEYS
        if extra_algo_keys:
            raise ValueError(f"Unknown algorithm settings detected: {extra_algo_keys}")

        if algo.get("n_components") != 1:
            raise ValueError("Contradictory configuration: n_components must equal 1 for this fixture")

        if len(raw_config["feature_ordering"]) != 2:
            raise ValueError("Contradictory feature ordering: expected 2 features [x1, x2]")


class PCA1DProjection:
    """1D PCA projection and reconstruction along a given unit principal component."""

    def __init__(self, unit_vector: Tuple[float, float], tolerance: float = 1e-5) -> None:
        # Normalize unit vector
        norm = math.sqrt(unit_vector[0] ** 2 + unit_vector[1] ** 2)
        if abs(norm - 1.0) > 1e-4:
            raise ValueError(f"Supplied vector is not a unit vector: norm = {norm}")
        self.v1 = unit_vector[0] / norm
        self.v2 = unit_vector[1] / norm
        self.tolerance = tolerance

    def project(self, points: List[List[float]]) -> List[float]:
        # Dot product with unit principal axis
        return [p[0] * self.v1 + p[1] * self.v2 for p in points]

    def reconstruct(self, projections: List[float]) -> List[List[float]]:
        # Reconstruct: projection * v
        return [[proj * self.v1, proj * self.v2] for proj in projections]

    def verify_reconstruction(self, points: List[List[float]]) -> Tuple[bool, float]:
        projections = self.project(points)
        reconstructed = self.reconstruct(projections)
        max_error = 0.0

        for original, rec in zip(points, reconstructed):
            err = math.sqrt((original[0] - rec[0]) ** 2 + (original[1] - rec[1]) ** 2)
            if err > max_error:
                max_error = err

        is_recovered = max_error <= self.tolerance
        return is_recovered, max_error


def load_fixture(path: Optional[Path] = None) -> Dict[str, Any]:
    target = path or (Path(__file__).resolve().parent / "fixture.json")
    with target.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline() -> Dict[str, Any]:
    payload = load_fixture()
    metadata = payload["metadata"]
    config = ReproduciblePCAConfig(metadata["config"])

    records = payload["records"]
    by_id = {r["record_id"]: [r["x1"], r["x2"]] for r in records}
    train_ids = config.split_ids["train"]
    points = [by_id[pid] for pid in train_ids]

    unit_vec = tuple(config.algorithm_settings["unit_vector"])
    pca = PCA1DProjection(unit_vector=unit_vec, tolerance=config.algorithm_settings["tolerance"])

    projections = pca.project(points)
    reconstructed = pca.reconstruct(projections)
    recovered, max_error = pca.verify_reconstruction(points)

    return {
        "task_id": "R41-T017",
        "schema_version": config.schema_version,
        "seed": config.seed,
        "n_points": len(points),
        "projections": [round(p, 4) for p in projections],
        "reconstructed": [[round(c, 4) for c in pt] for pt in reconstructed],
        "reconstruction_recovered": recovered,
        "max_reconstruction_error": round(max_error, 8)
    }


if __name__ == "__main__":
    result = run_pipeline()
    print(json.dumps(result, indent=2))
