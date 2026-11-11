# R41-T013: Compare a Minimal Baseline — MovieLens 100k – collaborative filtering

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic interaction fixture (`fixture.json`). These records are strictly for minimal baseline and pipeline verification—they are **not** observations from the real MovieLens 100k dataset and are **not** evidence of real-world or clinical performance.

## Data Contract & Synthetic Fixture

- **Schema:** `(user_id, item_id, rating)` with ordering attribute.
- **Fixture:** 6 interaction records across users `u1`, `u2`, `u3` and items `a`, `b`, `c`.
- **Split Policy:** Reserve the final interaction per user for a held-out evaluation fixture.
  - **Train Set:** `(u1, a, 5)`, `(u2, a, 4)`, `(u3, b, 4)`
  - **Held-Out Test Set:** `(u1, b, 3)`, `(u2, c, 2)`, `(u3, c, 5)`

## Baseline Strategy & Formula

- **Algorithm:** Training item-mean rating ($y_{i} = \frac{1}{|R_i|} \sum_{r \in R_i} r$), with fallback to global training mean for items not observed during training.
- **Train Parameters:** Item `a`: 4.5, Item `b`: 4.0, Global mean fallback: 4.3333.
- **Held-Out Evaluation:**
  - $u1 \rightarrow b$: actual 3.0, pred 4.0
  - $u2 \rightarrow c$: actual 2.0, pred 4.3333
  - $u3 \rightarrow c$: actual 5.0, pred 4.3333
  - **Metrics:** MAE $\approx$ 1.3333, MSE $\approx$ 2.2963

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t013 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Offline CPU-only bounded verification fixture.
- Any optional real-data extension must independently document dataset source, licence, access requirements, and chronological split methodology.
