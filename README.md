# NaturalImputation

**NaturalImputation** is a simple but effective target-driven imputation method for Logistic Regression. For a given feature $x$ and target $y$, let $\beta_0$ and $\beta_1$ be the coefficients that minimize cross-entropy for the relationship:

$$\ln\left(\frac{y'}{1-y'}\right) = \beta_0 + \beta_1 x'$$

Where $x'$ and $y'$ represent the subsets of $x$ and $y$ such that $x$ is non-missing. Then, $x$ is **naturally imputed** at the value:

$$x^{\ast} = \frac{\ln\left(\frac{y^{\ast}}{1-y^{\ast}}\right) - \beta_0}{\beta_1}$$

where $y^{\ast}$ represents the mean of the target variable for the missing observations.

This method avoids both the naïveté of mean/median imputation, and the complex dependence on other features inherent in more robust methods like MICE. It is effective when 1) $x'$ sufficiently predicts $y'$ (enforced automatically via a significance test on $\beta_1$) and 2) the target rate among missing observations diverges from the target rate among non-missing observations. When either condition fails, `impute_logistic` falls back to mean imputation.

## Example
```python
import numpy as np
import pandas as pd
from naturalimputation import impute_logistic
from sklearn.datasets import make_classification

rng = np.random.default_rng(0)
X, y = make_classification(
    random_state=rng.integers(0, 2**32 - 1)
)
X, y = pd.DataFrame(X), pd.Series(y)
test = rng.random(len(X)) < 0.25
impute_logistic(X[0], y, test)[:5]
```

```
0    0.259723
1   -1.231660
2    1.154356
3    0.464447
4   -0.106562
dtype: float64
```

## When does NaturalImputation help?

NaturalImputation exploits the difference between the target rate among missing vs non-missing observations. When that gap is large and the feature genuinely predicts the target, natural imputation delivers meaningful lift over mean imputation. `impute_logistic` automatically falls back to mean imputation when the feature's relationship with the target is not statistically significant (controlled by the `alpha` parameter, default 0.05).

The simulation below generates synthetic datasets with varying missingness patterns — including near-random missingness (`steepness=1.01`) — then bins each run by the average absolute target-rate gap ($|\\bar{y}\_{missing} - \\bar{y}\_{nonmissing}|$) and reports the mean AUC lift and the proportion of runs where NaturalImputation underperformed mean imputation:

```python
import warnings
import numpy as np
import pandas as pd
from naturalimputation.demo import run_experiment

warnings.filterwarnings("ignore")

results = []
for steepness in [1.01, 1.05, 1.1, 1.2, 1.5, 2, 3, 5]:
    mean_aucs, log_aucs, gaps = run_experiment(
        steepness=steepness,
        n_iterations=200,
        n_samples=10_000,
        flip_y=0.4,
        random_state=0,
    )
    for gap, lift in zip(gaps, log_aucs - mean_aucs):
        results.append({"target_rate_gap": gap, "lift": lift})

df = pd.DataFrame(results)
df["gap_bin"] = pd.qcut(df["target_rate_gap"], q=5)
summary = df.groupby("gap_bin")["lift"].agg(
    ["mean", "count", lambda x: (x < 0).mean()]
)
summary.columns = ["mean_lift", "count", "pct_negative"]
print(summary.round(4))
```

```
                       mean_lift  count  pct_negative
gap_bin
(0.012, 0.0228]           0.0053    320        0.2469
(0.0228, 0.0268]          0.0078    320        0.2969
(0.0268, 0.0316]          0.0096    320        0.3188
(0.0316, 0.041]           0.0193    320        0.2719
(0.041, 0.0912]           0.0181    320        0.4188
```

Lift increases with the target-rate gap. The top bin has the highest `pct_negative` because it includes runs where the gap is large by chance (random missingness) rather than by structure — the p-value guard catches most but not all of these cases.