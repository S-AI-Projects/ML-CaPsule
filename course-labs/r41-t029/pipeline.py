"""Content-based recommender and robustness verification for R41-T029.

Task: R41-T029 · 9 of 10
Project/section: Goodbooks-10k – content-based recommender
Student: Sharon Robinson
Repository: S-AI-Projects/ML-CaPsule

SYNTHETIC DATA NOTICE
=====================
All default interactions (u1..u3 x a..c) and item tags in this module are a
bounded synthetic fixture strictly for pipeline and interface robustness
verification. They are NOT observations from the Goodbooks-10k dataset, are
NOT evidence of real-world or clinical performance, and do NOT claim broad
adversarial robustness.

Stated Robustness Properties
----------------------------
1. **Valid transformation (preserves results after ID alignment)**:
   Reordering records across different users while strictly preserving each
   user's chronological ``user_seq`` order produces bit-identical train/val/test
   splits, user content profiles, held-out scores, and top-k recommendations
   once aligned by stable ``(user_id, item_id)`` IDs.
2. **Forbidden modality transformation (rejected)**:
   Randomizing or reversing within-user chronological order (``user_seq``
   decreasing for any user) raises ``TimeOrderViolationError``.
3. **Unsupported inputs (rejected or mapped to explicit fallback)**:
   - Missing required fields (``user_id``, ``item_id``, ``rating``) or invalid
     ratings raise ``SchemaValidationError``.
   - Unseen users, unseen items, or items whose content tokens are completely
     unseen map to explicit ``ScoreResult(is_fallback=True, fallback_reason=...)``
     entries rather than silently failing or fabricating similarities.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

# ── Constants ────────────────────────────────────────────────────────────────

TASK_ID: str = "R41-T029"
PROJECT_SECTION: str = "Goodbooks-10k – content-based recommender"
FIXED_SEED: int = 42
REQUIRED_INTERACTION_FIELDS: tuple[str, ...] = ("user_id", "item_id", "rating")
VALIDATION_HELD_OUT_USERS: frozenset[str] = frozenset({"u1", "u2"})
TEST_HELD_OUT_USERS: frozenset[str] = frozenset({"u3"})
MIN_RATING: float = 1.0
MAX_RATING: float = 5.0

LAB_DIR: Path = Path(__file__).resolve().parent
DEFAULT_FIXTURE_PATH: Path = LAB_DIR / "fixture.json"
DEFAULT_SCHEMA_PATH: Path = LAB_DIR / "schema.json"


# ── Exceptions ───────────────────────────────────────────────────────────────


class RecommenderRobustnessError(ValueError):
    """Base exception for R41-T029 recommender contract errors."""


class SchemaValidationError(RecommenderRobustnessError):
    """Raised when required fields, IDs, or ratings violate the data contract."""


class TimeOrderViolationError(RecommenderRobustnessError):
    """Raised when per-user chronological ordering is reversed or corrupted."""


class ModelNotFittedError(RuntimeError):
    """Raised when scoring or recommending before fitting."""


# ── Immutable Data Structures ────────────────────────────────────────────────


@dataclass(frozen=True)
class RecommenderSplits:
    """Explicit train/validation/test splits reserving each user's final interaction."""

    train: tuple[dict[str, Any], ...]
    validation: tuple[dict[str, Any], ...]
    test: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ScoreResult:
    """Deterministic score output with explicit fallback metadata."""

    user_id: str
    item_id: str
    score: float
    is_fallback: bool
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "item_id": self.item_id,
            "score": self.score,
            "is_fallback": self.is_fallback,
            "fallback_reason": self.fallback_reason,
        }


# ── Validation & Temporal Split Helpers ──────────────────────────────────────


def _validate_non_empty_str(value: Any, field_name: str, row_idx: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(
            f"Row {row_idx}: {field_name!r} must be a non-empty string, got {value!r}"
        )
    return value


def _validate_rating(value: Any, row_idx: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(
            f"Row {row_idx}: 'rating' must be numeric, got {value!r} ({type(value).__name__})"
        )
    rating = float(value)
    if not math.isfinite(rating) or rating < MIN_RATING or rating > MAX_RATING:
        raise SchemaValidationError(
            f"Row {row_idx}: 'rating' must be finite in [{MIN_RATING}, {MAX_RATING}], got {value!r}"
        )
    return rating


def validate_interactions(
    interactions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate schema fields, uniqueness, and monotonic per-user time order."""
    if not interactions:
        raise SchemaValidationError("Interactions must not be empty")

    validated: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    last_seq_by_user: dict[str, int] = {}

    for idx, row in enumerate(interactions):
        if not isinstance(row, Mapping):
            raise SchemaValidationError(f"Row {idx}: interaction must be a mapping")
        for field in REQUIRED_INTERACTION_FIELDS:
            if field not in row:
                raise SchemaValidationError(
                    f"Row {idx}: missing required field {field!r}"
                )
        user_id = _validate_non_empty_str(row["user_id"], "user_id", idx)
        item_id = _validate_non_empty_str(row["item_id"], "item_id", idx)
        rating = _validate_rating(row["rating"], idx)

        pair = (user_id, item_id)
        if pair in seen_pairs:
            raise SchemaValidationError(f"Row {idx}: duplicate (user_id, item_id) pair {pair!r}")
        seen_pairs.add(pair)

        user_seq = row.get("user_seq")
        if user_seq is not None:
            if isinstance(user_seq, bool) or not isinstance(user_seq, int) or user_seq < 1:
                raise SchemaValidationError(
                    f"Row {idx}: 'user_seq' must be a positive integer, got {user_seq!r}"
                )
            prev_seq = last_seq_by_user.get(user_id)
            if prev_seq is not None and user_seq <= prev_seq:
                raise TimeOrderViolationError(
                    f"Row {idx}: within-user time order violated for {user_id!r} "
                    f"(user_seq={user_seq} after {prev_seq})"
                )
            last_seq_by_user[user_id] = user_seq

        clean = dict(row)
        clean["user_id"] = user_id
        clean["item_id"] = item_id
        clean["rating"] = rating
        validated.append(clean)

    return validated


def split_leave_last_out(
    interactions: Sequence[Mapping[str, Any]],
    *,
    validation_users: frozenset[str] = VALIDATION_HELD_OUT_USERS,
    test_users: frozenset[str] = TEST_HELD_OUT_USERS,
) -> RecommenderSplits:
    """Reserve each user's final interaction for held-out validation/test splits.

    Earlier interactions for each user go to ``train``; each user's final
    interaction goes to ``validation`` or ``test`` deterministically by stable
    ``user_id``, keeping train/validation/test separation explicit and invariant
    to inter-user row ordering.
    """
    clean_rows = validate_interactions(interactions)
    by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        by_user[row["user_id"]].append(row)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []

    for user_id in sorted(by_user.keys()):
        u_rows = by_user[user_id]
        if len(u_rows) < 2:
            raise SchemaValidationError(
                f"User {user_id!r} has only {len(u_rows)} interaction(s); need >= 2 to hold out final interaction"
            )
        train_rows.extend(u_rows[:-1])
        final_row = u_rows[-1]
        if user_id in validation_users:
            val_rows.append(final_row)
        elif user_id in test_users:
            test_rows.append(final_row)
        else:
            test_rows.append(final_row)

    if not train_rows or not val_rows or not test_rows:
        raise SchemaValidationError("train, validation, and test splits must all be non-empty")

    return RecommenderSplits(
        train=tuple(train_rows),
        validation=tuple(val_rows),
        test=tuple(test_rows),
    )


def permute_preserving_user_time_order(
    interactions: Sequence[Mapping[str, Any]],
    *,
    seed: int = FIXED_SEED,
) -> list[dict[str, Any]]:
    """Valid modality-preserving perturbation: interleave users without reordering any user's timeline."""
    clean_rows = validate_interactions(interactions)
    queues: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean_rows:
        queues[row["user_id"]].append(dict(row))

    rng = random.Random(seed)
    active_users = sorted(queues.keys())
    permuted: list[dict[str, Any]] = []
    while active_users:
        chosen = rng.choice(active_users)
        permuted.append(queues[chosen].pop(0))
        if not queues[chosen]:
            active_users.remove(chosen)
    return permuted


# ── Bounded Content-Based Recommender ────────────────────────────────────────


def _cosine_similarity(vec_a: Mapping[str, float], vec_b: Mapping[str, float]) -> float:
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    dot = sum(val * vec_b.get(tok, 0.0) for tok, val in vec_a.items())
    return float(dot / (norm_a * norm_b))


class ContentBasedRecommender:
    """Offline CPU-only tag-cosine content-based recommender (Python 3 stdlib)."""

    def __init__(self, item_catalog: Mapping[str, Sequence[str]]) -> None:
        if not item_catalog:
            raise SchemaValidationError("item_catalog must not be empty")
        normalized: dict[str, dict[str, float]] = {}
        for item_id, tags in item_catalog.items():
            if not isinstance(item_id, str) or not item_id.strip():
                raise SchemaValidationError(f"Invalid catalog item_id: {item_id!r}")
            if not tags:
                raise SchemaValidationError(f"Item {item_id!r} must have at least one tag")
            vec: dict[str, float] = {}
            for tag in tags:
                if not isinstance(tag, str) or not tag.strip():
                    raise SchemaValidationError(f"Item {item_id!r} has invalid tag {tag!r}")
                vec[tag.lower()] = vec.get(tag.lower(), 0.0) + 1.0
            normalized[item_id] = vec

        self.item_vectors_: dict[str, dict[str, float]] = normalized
        self.user_profiles_: dict[str, dict[str, float]] = {}
        self.user_seen_items_: dict[str, frozenset[str]] = {}
        self.train_vocab_: frozenset[str] = frozenset()
        self.fallback_item_ranking_: tuple[str, ...] = ()
        self._is_fitted: bool = False

    def fit(self, train_interactions: Sequence[Mapping[str, Any]]) -> "ContentBasedRecommender":
        """Build rating-weighted user content profiles strictly from training interactions."""
        clean_train = validate_interactions(train_interactions)
        by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
        item_rating_sums: dict[str, float] = defaultdict(float)
        item_rating_counts: dict[str, int] = defaultdict(int)
        vocab: set[str] = set()

        for row in clean_train:
            item_id = row["item_id"]
            if item_id not in self.item_vectors_:
                raise SchemaValidationError(
                    f"Training interaction references unknown catalog item_id {item_id!r}"
                )
            by_user[row["user_id"]].append(row)
            item_rating_sums[item_id] += row["rating"]
            item_rating_counts[item_id] += 1
            vocab.update(self.item_vectors_[item_id].keys())

        profiles: dict[str, dict[str, float]] = {}
        seen_map: dict[str, frozenset[str]] = {}
        for user_id in sorted(by_user.keys()):
            u_rows = sorted(by_user[user_id], key=lambda r: (r["item_id"], r["rating"]))
            total_w = sum(r["rating"] for r in u_rows)
            profile: dict[str, float] = defaultdict(float)
            for row in u_rows:
                w = row["rating"] / total_w
                for tok, val in self.item_vectors_[row["item_id"]].items():
                    profile[tok] += w * val
            profiles[user_id] = dict(sorted(profile.items()))
            seen_map[user_id] = frozenset(r["item_id"] for r in u_rows)

        # Deterministic cold-start popularity fallback ordered by mean train rating desc, item_id asc
        def _fallback_sort_key(iid: str) -> tuple[float, str]:
            avg = (
                item_rating_sums[iid] / item_rating_counts[iid]
                if item_rating_counts[iid] > 0
                else 0.0
            )
            return (-avg, iid)

        self.user_profiles_ = profiles
        self.user_seen_items_ = seen_map
        self.train_vocab_ = frozenset(vocab)
        self.fallback_item_ranking_ = tuple(
            sorted(self.item_vectors_.keys(), key=_fallback_sort_key)
        )
        self._is_fitted = True
        return self

    def score_pair(
        self,
        user_id: str,
        item_id: str,
        *,
        override_item_tags: Sequence[str] | None = None,
    ) -> ScoreResult:
        """Score a (user_id, item_id) pair, mapping unseen entities to explicit fallbacks."""
        if not self._is_fitted:
            raise ModelNotFittedError("Call .fit() on training interactions before scoring")
        if not isinstance(user_id, str) or not user_id.strip():
            raise SchemaValidationError(f"user_id must be a non-empty string, got {user_id!r}")
        if not isinstance(item_id, str) or not item_id.strip():
            raise SchemaValidationError(f"item_id must be a non-empty string, got {item_id!r}")

        if user_id not in self.user_profiles_:
            return ScoreResult(
                user_id=user_id,
                item_id=item_id,
                score=0.0,
                is_fallback=True,
                fallback_reason="unseen_user",
            )

        if override_item_tags is not None:
            if not override_item_tags:
                raise SchemaValidationError("override_item_tags must not be empty")
            item_vec: dict[str, float] = {}
            for tag in override_item_tags:
                if not isinstance(tag, str) or not tag.strip():
                    raise SchemaValidationError(f"Invalid content tag: {tag!r}")
                item_vec[tag.lower()] = item_vec.get(tag.lower(), 0.0) + 1.0
            known_overlap = set(item_vec.keys()) & self.train_vocab_
            if not known_overlap:
                return ScoreResult(
                    user_id=user_id,
                    item_id=item_id,
                    score=0.0,
                    is_fallback=True,
                    fallback_reason="unseen_content_tokens",
                )
        else:
            if item_id not in self.item_vectors_:
                return ScoreResult(
                    user_id=user_id,
                    item_id=item_id,
                    score=0.0,
                    is_fallback=True,
                    fallback_reason="unseen_item",
                )
            item_vec = self.item_vectors_[item_id]

        sim = _cosine_similarity(self.user_profiles_[user_id], item_vec)
        return ScoreResult(
            user_id=user_id,
            item_id=item_id,
            score=sim,
            is_fallback=False,
            fallback_reason=None,
        )

    def recommend(self, user_id: str, *, top_k: int = 2) -> list[ScoreResult]:
        """Recommend unseen candidate items ordered by (-score, item_id), with explicit cold-start fallback."""
        if not self._is_fitted:
            raise ModelNotFittedError("Call .fit() on training interactions before recommending")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
            raise SchemaValidationError(f"top_k must be a positive int, got {top_k!r}")

        if user_id not in self.user_profiles_:
            return [
                ScoreResult(
                    user_id=user_id,
                    item_id=iid,
                    score=0.0,
                    is_fallback=True,
                    fallback_reason="unseen_user",
                )
                for iid in self.fallback_item_ranking_[:top_k]
            ]

        seen = self.user_seen_items_[user_id]
        candidates = [iid for iid in sorted(self.item_vectors_.keys()) if iid not in seen]
        scored = [self.score_pair(user_id, iid) for iid in candidates]
        scored.sort(key=lambda r: (-r.score, r.item_id))
        return scored[:top_k]


# ── Fixture Loading & End-to-End Robustness Pipeline ─────────────────────────


def load_fixture(path: Path | str | None = None) -> dict[str, Any]:
    """Load the synthetic six-interaction recommendation fixture."""
    fixture_path = Path(path) if path is not None else DEFAULT_FIXTURE_PATH
    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not data.get("is_synthetic", False):
        raise SchemaValidationError("Fixture must explicitly declare is_synthetic=true")
    return data


def load_schema(path: Path | str | None = None) -> dict[str, Any]:
    """Load the R41-T029 schema JSON."""
    schema_path = Path(path) if path is not None else DEFAULT_SCHEMA_PATH
    return json.loads(schema_path.read_text(encoding="utf-8"))


def evaluate_aligned_scores(
    model: ContentBasedRecommender,
    interactions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return scores keyed by stable 'user_id:item_id' for order-invariant comparison."""
    aligned: dict[str, dict[str, Any]] = {}
    for row in interactions:
        res = model.score_pair(row["user_id"], row["item_id"])
        key = f"{res.user_id}:{res.item_id}"
        aligned[key] = res.to_dict()
    return dict(sorted(aligned.items()))


def run_robustness_verification(
    fixture_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run original vs inter-user-permuted pipelines and explicit fallback checks."""
    fixture_doc = load_fixture(fixture_path)
    catalog = fixture_doc["item_catalog"]
    orig_interactions = fixture_doc["interactions"]
    permuted_interactions = permute_preserving_user_time_order(
        orig_interactions,
        seed=int(fixture_doc.get("seed", FIXED_SEED)),
    )

    orig_splits = split_leave_last_out(orig_interactions)
    perm_splits = split_leave_last_out(permuted_interactions)

    orig_model = ContentBasedRecommender(catalog).fit(orig_splits.train)
    perm_model = ContentBasedRecommender(catalog).fit(perm_splits.train)

    orig_val_scores = evaluate_aligned_scores(orig_model, orig_splits.validation)
    perm_val_scores = evaluate_aligned_scores(perm_model, perm_splits.validation)
    orig_test_scores = evaluate_aligned_scores(orig_model, orig_splits.test)
    perm_test_scores = evaluate_aligned_scores(perm_model, perm_splits.test)

    orig_recs = {
        uid: [r.to_dict() for r in orig_model.recommend(uid, top_k=2)]
        for uid in sorted(orig_model.user_profiles_.keys())
    }
    perm_recs = {
        uid: [r.to_dict() for r in perm_model.recommend(uid, top_k=2)]
        for uid in sorted(perm_model.user_profiles_.keys())
    }

    fallbacks = {
        "unseen_user": orig_model.score_pair("u_unknown", "a").to_dict(),
        "unseen_item": orig_model.score_pair("u1", "z_unknown").to_dict(),
        "unseen_content_tokens": orig_model.score_pair(
            "u1", "item_new", override_item_tags=["completely_unseen_tag"]
        ).to_dict(),
    }

    return {
        "task_id": TASK_ID,
        "project_section": PROJECT_SECTION,
        "verification_only": True,
        "claims_broad_adversarial_robustness": False,
        "synthetic_disclaimer": fixture_doc["disclaimer"],
        "seed": FIXED_SEED,
        "original_order_ids": [r["interaction_id"] for r in orig_interactions],
        "permuted_order_ids": [r["interaction_id"] for r in permuted_interactions],
        "splits": {
            "train": [(r["user_id"], r["item_id"], r["rating"]) for r in orig_splits.train],
            "validation": [(r["user_id"], r["item_id"], r["rating"]) for r in orig_splits.validation],
            "test": [(r["user_id"], r["item_id"], r["rating"]) for r in orig_splits.test],
        },
        "invariance_preserved": (
            orig_val_scores == perm_val_scores
            and orig_test_scores == perm_test_scores
            and orig_recs == perm_recs
        ),
        "validation_scores": orig_val_scores,
        "test_scores": orig_test_scores,
        "recommendations": orig_recs,
        "explicit_fallbacks": fallbacks,
    }
