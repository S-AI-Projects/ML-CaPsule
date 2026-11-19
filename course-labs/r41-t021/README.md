# R41-T021: Validate the Data Contract — MNIST – k-NN baseline

> **Pipeline Verification Notice:** All results in this lab use an offline, CPU-only synthetic fixture (`fixture.json`). These synthetic records are strictly for pipeline and data-contract verification—they are **not** observations from the named real MNIST dataset and are **not** evidence of real-world or clinical performance.

## Data Contract Schema

The dataset contract defines four required fields for every image record:

| Field | Type | Constraint | Description |
|---|---|---|---|
| `record_id` | `str` | Non-empty, unique | Unique identifier for the sample record |
| `group_id` | `str` | Non-empty | Group or cohort identifier (`grp_0`, `grp_1`) |
| `pixels` | `list[list[int\|float]]` | 4x4 matrix, $[0, 255]$ | 2D grayscale image pixel tensor |
| `label` | `int` | $\{0, 1\}$ | Binary ground-truth class label |

## Synthetic Fixture & Stratified Split

The fixture (`course-labs/r41-t021/fixture.json`) contains six synthetic 4×4 grayscale image rows:
- **Class 0:** 3 all-zero 4×4 images labelled `0` (`group_id: grp_0`).
- **Class 1:** 3 white diagonal (255) 4×4 images labelled `1` (`group_id: grp_1`).

### Stratified 1:1:1 Split Design

Train, validation, and test sets are strictly partitioned with 50/50 class balance:

| Split | Class 0 (Zero) | Class 1 (Diagonal) | Total |
|---|---|---|---|
| `train` | `rec_01` | `rec_04` | 2 |
| `validation` | `rec_02` | `rec_05` | 2 |
| `test` | `rec_03` | `rec_06` | 2 |

## Schema Validation Behavior

The validator (`validate_records` in `pipeline.py`) audits records without silently dropping rows:
1. **Missing Identifiers:** Detects absent `record_id` or `group_id` and records specific field failures with row index fallbacks.
2. **Duplicate Identifiers:** Flags repeated `record_id` entries with the first-observed row reference.
3. **Malformed Text:** Flags whitespace-only or non-string identifier fields.
4. **Malformed Numeric:** Disallows booleans masquerading as integers, non-integer labels, and labels outside $\{0, 1\}$.
5. **Malformed Tensor/Matrix:** Validates 4×4 matrix structure, detecting invalid row lengths, non-numeric pixel values, and out-of-range pixels ($<0$ or $>255$).

## Bounded CPU-only Baseline

The bundled `KNNImageClassifier` (with $k=1$) executes offline in milliseconds using only Python standard library:
- Fits on training representations (`rec_01`, `rec_04`).
- Evaluates on validation and test partitions, reaching 100% accuracy on the synthetic fixture.

## Automated Verification

```bash
python3 -m unittest discover -s course-labs/r41-t021 -p "test_*.py" -v
```

## Limits & Optional Real-Data Extension

- **Limits:** This is an offline, CPU smoke fixture designed strictly for pipeline verification and data contract assertions. It does not reflect complex handwritten digits.
- **Optional Real-Data Extension:** Any transition to the full MNIST dataset must separately document:
  - **Source:** Yann LeCun / Corinna Cortes / Christopher J.C. Burges (http://yann.lecun.com/exdb/mnist/) or OpenML `mnist_784`.
  - **Licence:** Creative Commons Attribution-Share Alike 3.0.
  - **Access Requirements:** Unrestricted public download via standard HTTPS / torchvision / scikit-learn.
  - **Split Design:** Canonical 60,000 training images and 10,000 test images with 28×28 pixel dimensions.
