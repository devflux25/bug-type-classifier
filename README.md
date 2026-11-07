# Bug Type Classifier

Predicts the type of a Python bug from a single buggy line of code, using TF-IDF features and classical machine learning on the PySStuBs dataset.

**Input:** one buggy line of Python. **Output:** one of 20 bug patterns, such as `Change Boolean Literal` or `Same Function More Args`.

## Dataset

[PySStuBs](https://zenodo.org/records/4589607) (Kamienski, Palechor, Bezemer, Hindle, University of Alberta): single-statement bugs mined from 1,000 popular open-source Python projects on GitHub. Licensed CC BY 4.0.

- 73,013 rows in the CSV; 44 rows with a missing `line_before` were dropped.
- The Zenodo page describes 23 patterns, but only 20 appear in the CSV.
- Classes are imbalanced: the largest (`Same Function More Args`) has 9,958 rows and the smallest (`Same Function Swap Args`) has 336.
- The data is not included in this repo. Download `pysstubs.csv` from the link above and place it in the project folder.

## Method

- **Input feature:** `line_before` only (the buggy line). `line_after` (the fix) is never used, because the model would not have it in real use.
- **Label:** `sstub_pattern`.
- **Features:** TF-IDF on character 2-4-grams (`analyzer="char_wb"`, `max_features=50000`). Character n-grams keep symbols like `+`, `(` and `.` that a word tokenizer would drop.
- **Split:** 80/20 with `GroupShuffleSplit` grouped by `project_name` (`random_state=42`), so no project appears in both train and test. Verified: zero overlapping projects. The test set has 13,619 rows.
- **Metrics:** accuracy and macro-F1. Macro-F1 averages over all 20 classes equally, so small classes count as much as large ones.

## Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Dummy (always the most frequent class) | 11.9% | 1.1% |
| Logistic Regression (`max_iter=1000`) | 38.1% | 29.8% |
| LinearSVC (`max_iter=5000`) | 36.6% | 29.9% |
| ComplementNB | 34.2% | 25.9% |
| Logistic Regression, `class_weight="balanced"` | 35.8% | **32.0%** |
| LinearSVC, `class_weight="balanced"` | 34.7% | 30.0% |

All models use the same features and the same train/test split. Class weighting was tried after seeing the first results, on the same test split.

## Findings

TODO (your words): which patterns the models handle best and worst, and why. Mention what the dummy baseline and the class_weight rows show.

## Limitations and next steps

TODO (your words): one line on the single split (differences of about 1 point are noise), one line on what extra information could help (for example, the surrounding code).

## Run it

Download `pysstubs.csv`, install `pandas` and `scikit-learn`, then run `python baseline.py`.