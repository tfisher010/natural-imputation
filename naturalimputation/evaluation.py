from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np


def evaluate(X, y, test):
    model = LogisticRegression(max_iter=1000).fit(X=X[~test], y=y[~test])
    preds = model.predict_proba(X)[:, 1]
    return np.array(
        [
            roc_auc_score(y_true=y[~test], y_score=preds[~test]),
            roc_auc_score(y_true=y[test], y_score=preds[test]),
        ]
    )
