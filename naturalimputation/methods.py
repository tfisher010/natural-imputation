import numpy as np
import pandas as pd
import statsmodels.api as sm


def impute_mean(x: np.ndarray):
    """Impute missing values in a 1D array with the column mean.

    Parameters
    ----------
    x : np.ndarray
        1D array with NaN values to impute.

    Returns
    -------
    tuple of (np.ndarray, float)
        Copy of x with NaN values replaced by the mean of non-missing values,
        and the imputation value used.
    """
    imp_val = np.nanmean(x)
    x_imp = x.copy()
    x_imp[np.isnan(x_imp)] = imp_val
    return x_imp, imp_val


def impute_naturally(x, y, test=None, alpha=0.05):
    """Impute missing values using the natural imputation method for logistic regression.

    Fits a logistic regression on observed (non-missing, non-test) data to model
    the relationship between x and y, then imputes missing values at the x value
    whose log-odds equals that of the mean target rate among missing observations.

    Falls back to mean imputation when the feature has no statistically significant
    relationship with the target (p > alpha on β₁), or when class diversity is
    insufficient to fit a model.

    Parameters
    ----------
    x : np.ndarray or pd.Series
        1D feature array with NaN values to impute.
    y : np.ndarray or pd.Series
        Binary target variable (0/1).
    test : np.ndarray or pd.Series, optional
        Boolean mask indicating test observations. Only training observations
        (where test is False) are used to fit the imputation model. If None,
        all observations are used for fitting.
    alpha : float, default 0.05
        Significance level for the β₁ coefficient. If the p-value exceeds
        alpha, the feature is considered non-predictive and mean imputation
        is used instead.

    Returns
    -------
    tuple of (np.ndarray or pd.Series, float)
        Copy of x with NaN values replaced, and the imputation value used.
        Returns a Series with the original index if x was a Series, otherwise
        returns an ndarray.
    """
    index = x.index if isinstance(x, pd.Series) else None
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    if test is None:
        test = np.zeros(len(x), dtype=bool)
    else:
        test = np.asarray(test)

    x_imp = x.copy()

    missing = np.isnan(x)
    if (len(np.unique(y[~test & ~missing])) < 2) or (
        len(np.unique(y[~test & missing])) < 2
    ):  # if only one (or zero) missing or nonmissing class, use mean imputation
        imp_val = np.nanmean(x)
        x_imp[missing] = imp_val
        result = pd.Series(x_imp, index=index) if index is not None else x_imp
        return result, imp_val
    train = ~test & ~missing
    X_train = sm.add_constant(x_imp[train])
    model = sm.Logit(y[train], X_train).fit(disp=0)
    beta_1 = model.params[1]
    p_value = model.pvalues[1]
    if p_value > alpha:  # if x does not significantly predict y, use mean imputation
        imp_val = np.nanmean(x)
        x_imp[missing] = imp_val
        result = pd.Series(x_imp, index=index) if index is not None else x_imp
        return result, imp_val
    null_success_rate = y[~test & missing].mean()
    imp_val = (
        np.log(null_success_rate / (1 - null_success_rate)) - model.params[0]
    ) / beta_1
    x_imp[missing] = imp_val
    result = pd.Series(x_imp, index=index) if index is not None else x_imp
    return result, imp_val
