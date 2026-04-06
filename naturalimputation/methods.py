import numpy as np
from sklearn.linear_model import LogisticRegression


def impute_mean(X: np.ndarray):
    X_imp = X.copy()
    for i in range(X.shape[1]):
        X_imp[np.isnan(X_imp[:, i]), i] = np.nanmean(X[:, i])
    return X_imp


def impute_logistic(x: np.ndarray, y: np.ndarray, test: np.ndarray):

    x_imp = x.copy()

    # raw_feature = X[:, i]
    missing = np.isnan(x)
    if (len(np.unique(y[~test & ~missing])) < 2) | len(
        np.unique(y[~test & missing])
    ) < 2:  # if only one (or zero) missing or nonmissing class, use mean imputation
        x_imp[missing] = np.nanmean(x)
        return x_imp
    model = LogisticRegression().fit(
        X=x_imp[~test & ~missing].reshape(-1, 1), y=y[~test & ~missing]
    )
    if model.coef_[0][0] == 0:  # if feature is unrelated to target, use mean imputation
        x_imp[missing] = np.nanmean(x)
        return x_imp
    null_success_rate = y[~test & missing].mean()
    imp_val = (
        np.log(null_success_rate / (1 - null_success_rate)) - model.intercept_[0]
    ) / model.coef_[0][0]
    x_imp[missing] = imp_val
    if np.isinf(imp_val) | np.isnan(imp_val):
        raise ValueError(
            null_success_rate,
            model.intercept_[0],
            model.coef_[0][0],
            y[~test & missing],
            len(x_imp[missing]),
            np.isnan(x_imp).any(),
        )
    return x_imp


# def impute_logistic(X: np.ndarray, y: np.ndarray, test: np.ndarray) -> np.ndarray:
#     X_imp = X.copy()
#     for i in range(X.shape[1]):
#         raw_feature = X[:, i]
#         missing = np.isnan(raw_feature)
#         if (len(np.unique(y[~test & ~missing])) < 2) | len(
#             np.unique(y[~test & missing])
#         ) < 2:  # if only one (or zero) missing or nonmissing class, use mean imputation
#             X_imp[missing, i] = np.nanmean(raw_feature)
#             continue
#         model = LogisticRegression().fit(
#             X=X_imp[~test & ~missing, i].reshape(-1, 1), y=y[~test & ~missing]
#         )
#         if (
#             model.coef_[0][0] == 0
#         ):  # if feature is unrelated to target, use mean imputation
#             X_imp[missing, i] = np.nanmean(raw_feature)
#             continue
#         null_success_rate = y[~test & missing].mean()
#         imp_val = (
#             np.log(null_success_rate / (1 - null_success_rate)) - model.intercept_[0]
#         ) / model.coef_[0][0]
#         X_imp[missing, i] = imp_val
#         if np.isinf(imp_val) | np.isnan(imp_val):
#             raise ValueError(
#                 null_success_rate,
#                 model.intercept_[0],
#                 model.coef_[0][0],
#                 y[~test & missing],
#                 len(X_imp[missing, i]),
#                 np.isnan(X_imp).any(),
#             )
#     return X_imp
