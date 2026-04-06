from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np


def evaluate(X, y, test):
    model = LogisticRegression().fit(X=X[~test], y=y[~test])
    return np.array(
        [
            roc_auc_score(y_true=y[~test], y_score=model.predict_proba(X)[:, 1][~test]),
            roc_auc_score(y_true=y[test], y_score=model.predict_proba(X)[:, 1][test]),
        ]
    )
