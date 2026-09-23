"""Preprocessing helpers shared across chapters."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .loading import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def make_preprocessor(
    numeric: list[str] | None = None,
    categorical: list[str] | None = None,
    scale: bool = True,
    impute: bool = True,
) -> ColumnTransformer:
    """Standard VoltGrid preprocessor.

    Scaling matters for LogisticRegression, SVM and KNN; it is harmless for
    trees. Keeping one preprocessor means a chapter can swap the estimator
    without rewriting the front of the pipeline.
    """
    numeric = numeric if numeric is not None else NUMERIC_FEATURES
    categorical = categorical if categorical is not None else CATEGORICAL_FEATURES

    num_steps = []
    if impute:
        num_steps.append(("impute", SimpleImputer(strategy="median")))
    if scale:
        num_steps.append(("scale", StandardScaler()))
    num_pipe = Pipeline(num_steps) if num_steps else "passthrough"

    cat_steps = []
    if impute:
        cat_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
    cat_steps.append(("onehot", OneHotEncoder(handle_unknown="ignore")))
    cat_pipe = Pipeline(cat_steps)

    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_xy(
    df: pd.DataFrame,
    target: str = "failed",
    features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a VoltGrid frame into X and y.

    Refuses to hand back columns that leak the target - see chapter 17.
    """
    LEAKY = {"error_code", "status", "session_id", target}
    if target == "failed":
        LEAKY.update({"energy_kwh", "duration_mins", "has_error_code"})
    if features is None:
        features = [c for c in df.columns if c != target and c not in LEAKY]
    else:
        bad = LEAKY.intersection(features)
        if bad:
            raise ValueError(
                f"These columns leak the target and must not be used as features: "
                f"{sorted(bad)}. See chapter 17 on leakage."
            )
    return df[features].copy(), df[target].copy()
