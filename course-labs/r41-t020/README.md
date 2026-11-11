# R41-T020: Define a Saved-Pipeline Interface: Mushroom Edibility – Naive Bayes

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic classification fixture (`fixture.json`). These records verify safe JSON-based pipeline serialization, reload prediction parity, and schema validation—they are **not** observations from the real Mushroom edibility dataset and are **not** evidence of real-world or clinical performance.

## Data Contract & Synthetic Fixture

- **Schema:** `(record_id, group_id, x1, x2, label)`
- **Fixture:** 8 synthetic rows ($x1 \in [0..7]$, $x2 = 1.0$, label $\in \{0, 1\}$).
- **Fixed Stratified Split:**
  - **Train:** `["r1", "r2", "r5", "r6"]`
  - **Validation:** `["r3", "r7"]`
  - **Test:** `["r4", "r8"]`

## Saved Pipeline Interface Guarantees

1. **Safe Serialization (No Untrusted Pickles):**
   - Pipeline configuration and parameters are serialized exclusively to JSON (`saved_pipeline.json`).
   - Arbitrary code execution vulnerabilities associated with `pickle` are eliminated.
2. **Zero Inference Credentials/Data Dependency:**
   - Model parameters (means, variances, class priors) load directly from the state dictionary.
   - No training data or remote credentials are required at inference time.
3. **Strict Prediction Parity:**
   - Predictions generated before saving match predictions after reloading identically.
4. **Schema & Version Validation:**
   - Validates `schema_version` and `feature_order` on load, raising `ValueError` on mismatches.
   - Validates feature presence, numeric types, and `record_id` during inference.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t020 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Offline CPU-only saved pipeline serialization and inference verification component only.
- Any optional real-data extension must separately record source, license, access requirements, and split design.
