# R41-T040: Define a Saved-Pipeline Interface — Wine quality – random-forest regression

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`) and safe JSON serialization. These synthetic records are strictly for pipeline interface verification—they are **not** observations from the named real Wine quality dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

- `record_id`: non-empty unique `str` (`r1`..`r6`)
- `x1`: finite scalar `float` in `{0, 1, 2, 3, 4, 5}`
- `x2`: constant feature scalar `float` (`1.0`)
- `target`: deterministic formula $2 \cdot x_1 + 1$
- `split`: split partition in `{"train", "validation", "test"}`

## Saved-Pipeline Interface Guarantees

1. **Security & Format:** Model configuration and parameters are serialized strictly via JSON (`save_json` / `load_json`), entirely avoiding untrusted pickle vulnerabilities.
2. **Reload Parity:** Predictions before saving and after deserialization agree within numerical tolerance ($10^{-10}$).
3. **Inference Autonomy:** Prediction requires zero training data, credentials, or network downloads.
4. **Schema Enforcement:** Input features missing expected fields (`x1`, `x2`) or non-finite tokens fail immediately with clear errors.
5. **Version Validation:** Loading states with mismatched schema versions triggers explicit incompatibility rejections.

## Verification Check

```bash
python3 -m unittest discover -s course-labs/r41-t040 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- Bounded offline CPU-only verification component only.
- Any optional real-data extension must separately record source, licence, access requirements, and split design.
