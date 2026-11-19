"""Fresh, unfitted pipelines for use with raw application DataFrames."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .feature_engineering import CATEGORICAL, NUMERIC, CreditFeatures


def build_preprocessor(*, scale_numeric=True):
    """Fit on a training fold, then transform other rows with that fold's parameters."""
    numbers = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler() if scale_numeric else "passthrough"),
    ])
    categories = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="(missing)", keep_empty_features=True)),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("numeric", numbers, NUMERIC),
        ("categorical", categories, CATEGORICAL),
    ])
    return Pipeline([("features", CreditFeatures()), ("columns", columns)])


def build_pipeline(estimator, *, scale_numeric=True):
    """Wrap Person 2's estimator so cross-validation clones all preprocessing steps."""
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=scale_numeric)),
        ("estimator", estimator),
    ])
