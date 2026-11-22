# R41-T029 · Goodbooks-10k – Content-Based Recommender: Robustness Fixture

**Task:** R41-T029 · 9 of 10 | **Student:** Sharon Robinson | **Branch:** `R41-T029` | **Assigned working branch:** `student/e26b07f0819`  
**Repository:** [S-AI-Projects/ML-CaPsule](https://github.com/S-AI-Projects/ML-CaPsule)

---

## Goal

Test a stated, modality-appropriate **robustness property** for the **Goodbooks-10k – content-based recommender** interface (`course-labs/r41-t029`):

- **Valid transformation (preserves results after ID alignment)**: Inter-user record-order permutation that **strictly preserves each user's chronological interaction order** (`user_seq` monotonic — time order is never randomized). After ID alignment by stable `(user_id, item_id)`, splits, user profiles, held-out scores, and top-k recommendations are bit-for-bit identical.
- **Forbidden modality transformation (rejected)**: Reversing/randomizing within-user time order raises `TimeOrderViolationError`.
- **Unsupported inputs (rejected or mapped to explicit fallback)**:
  - Missing required fields (`user_id`, `item_id`, `rating`) or invalid ratings raise `SchemaValidationError`.
  - Unseen users (`unseen_user`), unseen items (`unseen_item`), and items with all-unseen content tokens (`unseen_content_tokens`) map to explicit `ScoreResult(is_fallback=True, score=0.0, fallback_reason=...)`.

> **SYNTHETIC DATA & SCOPE DISCLAIMER:** All six interactions and item tags in `fixture.json` are a **synthetic pipeline-verification fixture** — NOT observations from Goodbooks-10k and NOT evidence of real-world or clinical performance. These tests verify the stated interface invariants only and do **not** claim broad adversarial robustness.

---

## Data Contract

Defined in [`schema.json`](schema.json) and instantiated literally in [`fixture.json`](fixture.json).

### Literal Synthetic Interactions (Leave-Last-Out Split)

| interaction_id | user_id | item_id | rating | user_seq | Split |
|----------------|---------|---------|--------|----------|-------|
| `i1` | `u1` | `a` | `5` | `1` | `train` |
| `i2` | `u1` | `b` | `3` | `2` | `validation` *(final for `u1`)* |
| `i3` | `u2` | `a` | `4` | `1` | `train` |
| `i4` | `u2` | `c` | `2` | `2` | `validation` *(final for `u2`)* |
| `i5` | `u3` | `b` | `4` | `1` | `train` |
| `i6` | `u3` | `c` | `5` | `2` | `test` *(final for `u3`)* |

Synthetic item catalog: `a -> ["fantasy", "adventure"]`, `b -> ["fantasy", "mystery"]`, `c -> ["science", "mystery"]`.

---

## Pipeline Verification Results (Synthetic Fixture Only)

- **Original interaction order**: `["i1", "i2", "i3", "i4", "i5", "i6"]`
- **Valid inter-user permuted order (`seed=42`, per-user `user_seq=[1, 2]` preserved)**: `["i5", "i1", "i2", "i6", "i3", "i4"]`
- **Invariance preserved after `(user_id, item_id)` alignment**: `True`

| Split | Pair (`user_id:item_id`) | Original Score | Permuted Score | Fallback? |
|-------|--------------------------|----------------|----------------|-----------|
| `validation` | `u1:b` | `0.5` | `0.5` | `False` |
| `validation` | `u2:c` | `0.0` | `0.0` | `False` |
| `test` | `u3:c` | `0.5` | `0.5` | `False` |

Top-2 unseen recommendations (identical before and after permutation): `u1 -> [b (0.5), c (0.0)]`, `u2 -> [b (0.5), c (0.0)]`, `u3 -> [a (0.5), c (0.5)]`.

Explicit fallbacks verified:
- `("u_unknown", "a")` -> `is_fallback=True`, `fallback_reason="unseen_user"`, `score=0.0`
- `("u1", "z_unknown")` -> `is_fallback=True`, `fallback_reason="unseen_item"`, `score=0.0`
- `("u1", "item_new", ["completely_unseen_tag"])` -> `is_fallback=True`, `fallback_reason="unseen_content_tokens"`, `score=0.0`

---

## Running the Automated Check

Uses only the Python 3 standard library (offline, CPU-only, zero external dependencies):

```bash
python3 -m unittest discover -s course-labs/r41-t029 -p "test_*.py" -v
```

### Unit Test Coverage (12 tests)

- `TestSyntheticInteractionFixtureContract` (3): synthetic & non-adversarial scope notices, literal 6 interactions, leave-last-out disjoint train/validation/test split
- `TestValidTransformationRobustness` (3): inter-user permutation differs in global row order yet preserves aligned scores & rankings, exact documented scores/rankings, within-user `user_seq` monotonicity
- `TestUnsupportedInputsAndExplicitFallbacks` (5): reversed within-user time order rejected, missing required fields rejected, invalid ratings rejected, explicit fallbacks for unseen user/item/tokens, unfitted error
- `TestDeliberateMutationGuard` (1): proves altered expected held-out score triggers `AssertionError`

---

## Limits

- Verifies **bounded interface robustness properties only** (order invariance under valid inter-user interleaving, time-order rejection, missing-field rejection, explicit cold-start/OOV fallbacks) — explicitly does **not** claim broad adversarial robustness (e.g., shilling attacks or poisoning resistance).
- Uses 6 synthetic interactions and 3 synthetic tag pairs; carries zero statistical evidence of real Goodbooks-10k recommendation quality (NDCG, Recall@K, coverage).

---

## Optional Real-Data Extension Contract

Any future extension on real book recommendation data **must separately document** its provenance and distinguish real-data metrics from synthetic pipeline-verification outputs:

| Requirement | Specification |
|-------------|---------------|
| **Source** | Goodbooks-10k dataset (Zygmunt Zając, 2017) |
| **Licence** | CC BY-SA 4.0 (verify upstream repository licence before redistribution) |
| **Access requirements** | Publicly available via https://github.com/zygmuntz/goodbooks-10k (no paid service required; keep raw CSVs out of Git per `.gitignore`) |
| **Split design** | ~6M ratings across 53,424 users and 10,000 books with book tags (`book_tags.csv`, `tags.csv`); use chronological or per-user leave-last-out train/validation/test splits without leaking future interactions or test book tag statistics into user profiles |

---

## Tool Versions

- **Runtime requirement**: Python 3 standard library only (verified offline on Python `3.15.2`, macOS arm64)
