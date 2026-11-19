# FinBridge Capital credit risk project

Our group is trying to answer one question. When someone applies for a loan, how likely are they to default? We can only use information that would be available when they apply.

My part (Person 1) was to explore the data, decide which columns we could use, and write the steps that prepare them for a model. Person 2 can use those steps with their own model.

## What's in the project?

| File or folder | What you'll find |
|---|---|
| `train.csv` and `test.csv` | One row per application. The 31,016 training rows include `default_flag`: 0 means repaid, and 1 means defaulted or charged off. The 7,754 test rows have the same 22 input columns, with the outcome left out. |
| Assignment PDF | The project brief, column descriptions, and final submission rules. Keep it and both CSVs in the main project folder. |
| `notebooks/` | Three notebooks to read in order. **01** checks the files and their columns. **02** explores missing values and patterns in defaults. **03** looks at the new columns I calculated. |
| `src/` | `feature_engineering.py` selects columns and calculates new ones. `preprocessing.py` puts the preparation steps together so we can reuse them. |
| `reports/` | `eda_findings.md` explains what I found. `data_quality_report.csv` lists missing and distinct values for each training column. `feature_documentation.md` explains each feature choice. The charts are in `figures/`. |
| `tests/test_preprocessing.py` | Checks how the code handles missing values, new categories, and training and test rows. It also checks that the original CSV files are intact. |
| `requirements.txt` and `.gitignore` | The first lists the Python libraries and versions to install. The second keeps local files such as the Python environment out of Git. |

## How to run it

**1. Set up Python.** Install Python 3.12 and open a terminal in the folder containing `train.csv`. Create a virtual environment, which keeps this project's libraries together. On Windows, run these lines in PowerShell.

```powershell
py -3.12 -m venv .venv
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
```

On macOS or Linux, use these two lines instead.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**2. Install the libraries.** In the same terminal, run:

```sh
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

**3. Run the notebooks.** Start JupyterLab with `python -m jupyterlab`. Open the `notebooks` folder, select the project's Python environment if asked, and choose **Run → Run All Cells** in each notebook, in order from 01 to 03.

You can also run them from the terminal. These commands save the results inside each notebook and refresh the reports and charts.

```sh
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 notebooks/01_dataset_understanding.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 notebooks/02_exploratory_data_analysis.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=180 notebooks/03_feature_engineering.ipynb
```

**4. Run the checks.** From the main project folder, run `python -m pytest -q`.

## What I did with the data

**Exploratory data analysis (EDA)** means looking at the data to see what's there. For example, I counted 4,507 defaults among 31,016 training applications (14.53%). I also checked unusually large values. The highest reported annual income was $6 million. I kept that row because the files give us no way to tell whether the amount is an error.

**Preprocessing** turns each application into numbers a model can use. In training, 38 rows were missing credit utilization (the percentage of available revolving credit used). The code fills a missing number with the middle value, or median, from the rows used for training. For the 829 missing employment lengths, it keeps a separate “missing” category. It turns text categories into yes/no columns and can handle a category it hasn't seen before.

**Feature engineering** means calculating a new column from information we already have. I added three:

- `loan_to_income` divides the loan request by annual income. A $10,000 request on a $50,000 income gives 0.20, or 20% of one year's income.
- `revolving_balance_to_income` divides the existing revolving credit balance by annual income. It shows how large that balance is compared with earnings.
- `has_prior_delinquency` is 1 if the borrower had a recorded delinquency in the past two years and 0 if the count was zero. A missing count stays missing until preprocessing.

If income is missing, zero, or negative, the two ratios are treated as missing.

## Columns I left out

- `application_id` and `applicant_ref_id` identify records. Keep the test application IDs in their original order so predictions can be matched to the right loans.
- `default_flag` is the answer the model is trying to predict.
- `recent_payment_amount` describes money already received. We can't confirm it was known when the borrower applied.
- The assigned interest rate, scheduled term and monthly payment, risk tier and subtier, and income verification status may also come from later checks. The files don't say when those values were set. For the credit history fields we do use, we assume they describe the borrower at the time of application.

## How Person 2 can use it

A **pipeline** runs the same preparation steps each time and then passes the results to your model. Here's how to connect a scikit-learn model from a script in the main project folder. Logistic regression is just an example; you can put your own model inside `build_pipeline()`.

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.preprocessing import build_pipeline

train = pd.read_csv("train.csv")
X = train.drop(columns="default_flag")
y = train["default_flag"]

pipeline = build_pipeline(LogisticRegression(max_iter=1000))
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y, cv=folds, scoring="roc_auc")
print(scores)
```

This check is called **cross-validation**. It takes turns holding out one of five groups of rows, keeping a similar share of defaults in each group. ROC-AUC measures how well the model ranks borrowers who default above borrowers who repay.

Pass the **raw rows** to the pipeline. That way, each round learns its medians, categories, and number scales from its own training rows. Preparing the whole dataset first would let information from the held-out rows slip into training.

If you only want to inspect the prepared columns, call `prep = build_preprocessor()` after importing it from `src.preprocessing`. Use `prep.fit_transform(X_train)` on your training split, then `prep.transform(X_valid)` or `prep.transform(test)` on the other rows. Use `prep.get_feature_names_out()` to see the column names. Both pipeline builders accept `scale_numeric=False` if your model doesn't need numbers put on a common scale.
