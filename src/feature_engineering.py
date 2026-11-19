"""Application and bureau features with no learned statistics."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


RAW_NUMERIC = [
    "requested_amount", "annual_income", "debt_to_income_ratio", "delinquencies_24m",
    "credit_inquiries_6m", "active_credit_lines", "derogatory_public_records",
    "revolving_credit_balance", "revolving_utilization_pct", "total_credit_accounts",
]
CATEGORICAL = ["employment_tenure", "housing_status", "loan_purpose"]
ENGINEERED = ["loan_to_income", "revolving_balance_to_income", "has_prior_delinquency"]
NUMERIC = RAW_NUMERIC + ENGINEERED


def as_number(series, *, percent=False):
    """Parse a numeric column, allowing a percent suffix only for percentage fields."""
    if not pd.api.types.is_numeric_dtype(series):
        series = series.astype("string").str.strip().replace("", pd.NA)
        if percent:
            series = series.str.removesuffix("%")
    return pd.to_numeric(series, errors="raise").astype(float).replace([np.inf, -np.inf], np.nan)


class CreditFeatures(TransformerMixin, BaseEstimator):
    """Select explicitly allowed columns before computing row-wise features."""

    def fit(self, X, y=None):
        self._validate_columns(X)
        self.is_fitted_ = True
        return self

    @staticmethod
    def _validate_columns(X):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("Supply a pandas DataFrame with unique column names.")
        missing = sorted(set(RAW_NUMERIC + CATEGORICAL) - set(X.columns))
        if missing:
            raise ValueError(f"Missing application columns: {missing}")

    def transform(self, X):
        check_is_fitted(self, "is_fitted_")
        self._validate_columns(X)
        result = pd.DataFrame(index=X.index)
        for name in RAW_NUMERIC:
            result[name] = as_number(X[name], percent=(name == "revolving_utilization_pct"))
        income = result["annual_income"].where(result["annual_income"] > 0)
        result["loan_to_income"] = result["requested_amount"] / income
        result["revolving_balance_to_income"] = result["revolving_credit_balance"] / income
        prior = result["delinquencies_24m"]
        result["has_prior_delinquency"] = prior.gt(0).astype(float).where(prior.notna())
        result = result.replace([np.inf, -np.inf], np.nan)
        for name in CATEGORICAL:
            clean = X[name].astype("string").str.strip().replace("", pd.NA)
            result[name] = clean.to_numpy(dtype=object, na_value=np.nan)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(NUMERIC + CATEGORICAL, dtype=object)
