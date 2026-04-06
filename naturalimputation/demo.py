import numpy as np
from .evaluation import evaluate
from joblib import Parallel, delayed
from .methods import impute_mean, impute_naturally
from numpy.random import SeedSequence, default_rng
from sklearn.datasets import make_classification
from scipy.stats import beta


def generate_dataset(steepness: int = 5, random_state=None, **classification_kwargs):
    rng = np.random.default_rng(random_state)
    X, y = make_classification(
        random_state=rng.integers(0, 2**32 - 1), **classification_kwargs
    )
    for i in range(X.shape[1]):
        a, b = (1, steepness) if rng.random() > 0.5 else (steepness, 1)
        x_min, x_max = X[:, i].min(), X[:, i].max()
        x_scaled = (X[:, i] - x_min) / (x_max - x_min)
        probs = beta.pdf(x_scaled, a, b)
        probs_normalized = probs / probs.max()
        mask = probs_normalized < rng.random(size=X.shape[0])
        X[mask, i] = np.nan
    test = rng.random(X.shape[0]) < 0.25
    return X, y, test


def run_simulation(child_seed, steepness, **kwargs):
    rng = default_rng(child_seed)
    X, y, test = generate_dataset(random_state=rng, steepness=steepness, **kwargs)

    # compute average target-rate gap across features
    gaps = []
    for i in range(X.shape[1]):
        missing = np.isnan(X[:, i])
        train_missing = ~test & missing
        train_nonmissing = ~test & ~missing
        if train_missing.any() and train_nonmissing.any():
            gaps.append(abs(y[train_missing].mean() - y[train_nonmissing].mean()))
    avg_gap = np.mean(gaps) if gaps else 0.0

    X_mean = X.copy()
    for i in range(X.shape[1]):
        X_mean[:, i], _ = impute_mean(X[:, i])
    res_mean = evaluate(X_mean, y, test)

    X_imp = X.copy()
    for i in range(X.shape[1]):
        X_imp[:, i], _ = impute_naturally(X[:, i], y, test)
    res_log = evaluate(X_imp, y, test)
    return res_mean[1], res_log[1], avg_gap


def run_experiment(n_iterations, steepness=5, random_state=42, n_jobs=-1, **kwargs):
    ss = SeedSequence(random_state)
    child_seeds = ss.spawn(n_iterations)
    results = Parallel(n_jobs=n_jobs)(
        delayed(run_simulation)(seed, steepness, **kwargs) for seed in child_seeds
    )
    mean_aucs, log_aucs, gaps = zip(*results)
    return np.array(mean_aucs), np.array(log_aucs), np.array(gaps)
