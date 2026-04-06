import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def impute_mean(x: np.ndarray):
    x_imp = x.copy()
    x_imp[np.isnan(x_imp)] = np.nanmean(x)
    return x_imp


def impute_logistic(x, y, test):
    index = x.index if isinstance(x, pd.Series) else None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    test = np.asarray(test)

    x_imp = x.copy()

    missing = np.isnan(x)
    if (len(np.unique(y[~test & ~missing])) < 2) or (
        len(np.unique(y[~test & missing])) < 2
    ):  # if only one (or zero) missing or nonmissing class, use mean imputation
        x_imp[missing] = np.nanmean(x)
        return pd.Series(x_imp, index=index) if index is not None else x_imp
    model = LogisticRegression().fit(
        X=x_imp[~test & ~missing].reshape(-1, 1), y=y[~test & ~missing]
    )
    if model.coef_[0][0] == 0:  # if feature is unrelated to target, use mean imputation
        x_imp[missing] = np.nanmean(x)
        return pd.Series(x_imp, index=index) if index is not None else x_imp
    null_success_rate = y[~test & missing].mean()
    imp_val = (
        np.log(null_success_rate / (1 - null_success_rate)) - model.intercept_[0]
    ) / model.coef_[0][0]
    x_imp[missing] = imp_val
    return pd.Series(x_imp, index=index) if index is not None else x_imp
