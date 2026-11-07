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

The models are best at bugs that leave a visible trace in the buggy line. With balanced logistic regression, the top patterns are Change Boolean Literal (F1 0.58), Change Numeric Literal (0.57), Change Operand (0.55), Add Elements to Iterable (0.52) and Change Unary Operator (0.51). They fail on bugs where something is missing or rearranged, since the line looks like normal code: Same Function Swap Args (0.01), Add Attribute Access (0.04) and Same Function Wrong Caller (0.10). Swap Args has only 53 test rows, so its score is noisy, but Add Attribute Access has 282 rows and still scores near zero.

Four very different setups scored within a few points of each other, so the limit is probably the information in one line and not the choice of model. Class weighting raised logistic regression's macro-F1 from 29.8% to 32.0% and its recall on rare classes (for example, Change Binary Operator went from 0.06 to 0.25), but accuracy fell from 38.1% to 35.8%, because the model now guesses rare classes more often and is wrong more often. For LinearSVC, class weighting changed macro-F1 by only 0.1 points and lowered accuracy.

## Limitations and next steps

All results come from a single project-grouped split, so differences of about 1 point (such as LogReg vs LinearSVC) should be treated as noise, and the class-weighting runs were chosen after seeing the first test scores. The next step is to give the model more context than one line, such as the surrounding code, and to repeat the comparison with grouped cross-validation instead of one split.

## Run it

Download `pysstubs.csv`, install `pandas` and `scikit-learn`, then run `python baseline.py`.