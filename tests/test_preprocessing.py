"""Regression checks against the supplied raw files and controlled fold inputs."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_validate
from sklearn.pipeline import Pipeline

from src.feature_engineering import CreditFeatures, NUMERIC
from src.preprocessing import build_pipeline, build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = [
    "application_id", "applicant_ref_id", "default_flag", "recent_payment_amount",
    "risk_tier", "risk_subtier", "monthly_emi", "interest_rate_pct",
    "repayment_tenure", "income_verification",
]


@pytest.fixture(scope="module")
def folds():
    data = pd.read_csv(ROOT / "train.csv")
    split = StratifiedShuffleSplit(n_splits=1, train_size=1000, test_size=400, random_state=42)
    train, valid = next(split.split(data, data.default_flag))
    return data.iloc[train], data.iloc[valid]


def test_train_validation_and_full_test(folds):
    train, valid = folds
    prep = build_preprocessor()
    training = prep.fit_transform(train)
    assert training.shape[1] == len(prep.get_feature_names_out())
    assert np.isfinite(training).all()
    for frame in [valid, pd.read_csv(ROOT / "test.csv")]:
        result = prep.transform(frame)
        assert result.shape == (len(frame), training.shape[1])
        assert np.isfinite(result).all()


def test_excluded_columns_cannot_change_predictions_inputs(folds):
    train, valid = folds
    prep = build_preprocessor().fit(train)
    assert not any(col in name for name in prep.get_feature_names_out() for col in EXCLUDED)
    np.testing.assert_array_equal(prep.transform(valid), prep.transform(valid.drop(columns=EXCLUDED)))


def test_ratios_and_undefined_inputs(folds):
    train, valid = folds
    rows = valid.iloc[:4].copy()
    rows["annual_income"] = [12000., 0., np.nan, np.inf]
    rows["requested_amount"] = 6000.
    rows["revolving_credit_balance"] = 1200.
    rows["delinquencies_24m"] = [0., 2., np.nan, 1.]
    result = CreditFeatures().fit(train).transform(rows)
    assert result.index.equals(rows.index)
    np.testing.assert_allclose(result.loan_to_income, [.5, np.nan, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(result.revolving_balance_to_income, [.1, np.nan, np.nan, np.nan], equal_nan=True)
    np.testing.assert_allclose(result.has_prior_delinquency, [0, 1, np.nan, 1], equal_nan=True)
    assert np.isfinite(build_preprocessor().fit(train).transform(rows)).all()


def test_fold_statistics_and_unknown_category(folds):
    train, valid = (frame.copy() for frame in folds)
    train["annual_income"] = 40000.
    train.iloc[0, train.columns.get_loc("annual_income")] = np.nan
    valid["annual_income"] = 1000000.
    valid["housing_status"] = "unseen validation category"
    prep = build_preprocessor().fit(train)
    result = prep.transform(valid)
    numeric = prep.named_steps["columns"].named_transformers_["numeric"]
    income = NUMERIC.index("annual_income")
    assert numeric.named_steps["impute"].statistics_[income] == 40000.
    assert numeric.named_steps["scale"].mean_[income] == 40000.
    housing_cols = [i for i, name in enumerate(prep.get_feature_names_out()) if "housing_status_" in name]
    assert housing_cols and (result[:, housing_cols] == 0).all()
    assert np.isfinite(result).all()


def test_estimator_wrapper_and_optional_scaling(folds):
    train, _ = folds
    # A passthrough last step exercises the wrapper without training a classifier.
    wrapped = clone(build_pipeline("passthrough", scale_numeric=False))
    assert isinstance(wrapped, Pipeline)
    result = wrapped.fit_transform(train)
    np.testing.assert_allclose(result[:, NUMERIC.index("annual_income")], train.annual_income)


@pytest.mark.parametrize(("name", "digest"), [
    ("train.csv", "7e706e323443ed2fbe6c2acd73265fcdf13c6a9435e492ecc2be29774fdfadfd"),
    ("test.csv", "e96a1fdc1ae2f3d6066df0e3b578c2523f4d2918c8bf5ffabd1dd0a633b6f8b0"),
])
def test_original_file_bytes(name, digest):
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest


def test_numeric_units_and_missing_categories(folds):
    train, valid = folds
    rows = valid.iloc[:2].copy()
    rows["revolving_utilization_pct"] = [" 76.7% ", " "]
    rows["employment_tenure"] = [" ", None]
    prep = build_preprocessor(scale_numeric=False).fit(train)
    values = pd.DataFrame(prep.transform(rows), columns=prep.get_feature_names_out())
    training_utilization = train.revolving_utilization_pct.str.rstrip("%").astype(float)
    np.testing.assert_allclose(values["numeric__revolving_utilization_pct"],
                               [76.7, training_utilization.median()])
    assert (values["categorical__employment_tenure_(missing)"] == 1).all()
    rows["annual_income"] = ["120%", "12000"]
    with pytest.raises(ValueError):
        prep.transform(rows)


def test_cross_validation_fits_each_fold_separately(folds):
    train, _ = folds
    X = train.drop(columns="default_flag").copy()
    # Unique labels make validation-only categories identifiable in every split.
    X["housing_status"] = X.index.astype(str)
    y = train.default_flag
    splits = list(StratifiedKFold(3, shuffle=True, random_state=42).split(X, y))

    def check_validation_output(pipeline, validation, labels):
        return float(np.isfinite(pipeline.transform(validation)).all())

    run = cross_validate(build_pipeline("passthrough"), X, y, cv=splits,
                         scoring=check_validation_output, return_estimator=True, error_score="raise")
    assert (run["test_score"] == 1).all()
    for (fit_rows, _), pipeline in zip(splits, run["estimator"]):
        fold = X.iloc[fit_rows]
        columns = pipeline.named_steps["preprocess"].named_steps["columns"]
        numbers = columns.named_transformers_["numeric"]
        income = NUMERIC.index("annual_income")
        assert numbers.named_steps["impute"].statistics_[income] == fold.annual_income.median()
        np.testing.assert_allclose(numbers.named_steps["scale"].mean_[income], fold.annual_income.mean())
        encoder = columns.named_transformers_["categorical"].named_steps["encode"]
        fields = columns.transformers_[1][2]
        vocab = dict(zip(fields, encoder.categories_))
        assert set(vocab["housing_status"]) == set(fold.housing_status)
