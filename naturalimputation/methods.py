import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def impute_mean(x: np.ndarray):
    """Impute missing values in a 1D array with the column mean.

    Parameters
    ----------
    x : np.ndarray
        1D array with NaN values to impute.

    Returns
    -------
    np.ndarray
        Copy of x with NaN values replaced by the mean of non-missing values.
    """
    x_imp = x.copy()
    x_imp[np.isnan(x_imp)] = np.nanmean(x)
    return x_imp


def impute_logistic(x, y, test):
    """Impute missing values using the natural imputation method for logistic regression.

    Fits a logistic regression on observed (non-missing, non-test) data to model
    the relationship between x and y, then imputes missing values at the x value
    whose log-odds equals that of the mean target rate among missing observations.

    Falls back to mean imputation when the feature has no predictive relationship
    with the target, or when class diversity is insufficient to fit a model.

    Parameters
    ----------
    x : np.ndarray or pd.Series
        1D feature array with NaN values to impute.
    y : np.ndarray or pd.Series
        Binary target variable (0/1).
    test : np.ndarray or pd.Series
        Boolean mask indicating test observations. Only training observations
        (where test is False) are used to fit the imputation model.

    Returns
    -------
    np.ndarray or pd.Series
        Copy of x with NaN values replaced. Returns a Series with the original
        index if x was a Series, otherwise returns an ndarray.
    """
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
