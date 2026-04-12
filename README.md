# NaturalImputation

**NaturalImputation** is a target-driven imputation method for Logistic Regression. It imputes each missing feature at the value whose predicted log-odds matches missing rows' observed target rate.

## How it works

For a given feature $x$ and binary target $y$, fit a logistic regression on the non-missing observations:

$$\ln\left(\frac{y'}{1-y'}\right) = \beta_0 + \beta_1 x'$$

Then impute missing values of $x$ at:

$$x^{\ast} = \frac{\ln\left(\frac{y^{\ast}}{1-y^{\ast}}\right) - \beta_0}{\beta_1}$$

where $y^{\ast}$ is the mean target rate among missing observations.

## Why use it?

NaturalImputation is univariate: each feature is imputed independently using only the target, hence no risk of multicollinearity and trivial parallelization, unlike multivariate methods like MICE.

It is effective when 1) the feature sufficiently predicts the target (enforced automatically via a significance test on $\beta_1$) and 2) the target rate among missing observations diverges from the rate among non-missing observations. When either condition fails, `impute_naturally` falls back to mean imputation.

## Example
```python
import numpy as np
import pandas as pd
from naturalimputation import impute_naturally
from sklearn.datasets import make_classification

rng = np.random.default_rng(0)
X, y = make_classification(
    random_state=rng.integers(0, 2**32 - 1)
)
X, y = pd.DataFrame(X), pd.Series(y)
test = rng.random(len(X)) < 0.25
x_imputed, imp_val = impute_naturally(X[0], y, test)
print(f"Imputation value: {imp_val:.4f}")
print(x_imputed[:5])
```

```
Imputation value: 0.0405
0    0.259723
1   -1.231660
2    1.154356
3    0.464447
4   -0.106562
dtype: float64
```

## When does NaturalImputation help?

NaturalImputation exploits the difference between the target rate among missing vs non-missing observations. When that gap is large and the feature predicts the target, natural imputation delivers meaningful lift over mean imputation. `impute_naturally` automatically falls back to mean imputation when the feature's relationship with the target is not statistically significant (controlled by the `alpha` parameter, default 0.05).

The below simulation generates synthetic datasets with varying missingness patterns, including near-random missingness (`steepness=1.01`), then bins each run by the average absolute target-rate gap ($|\\bar{y}\_{missing} - \\bar{y}\_{nonmissing}|$) and reports the mean AUC lift and the proportion of runs where NaturalImputation underperformed mean imputation:

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

NaturalImputation generally delivers more lift when the target-rate gap is large. The top bin has the highest `pct_negative` because it includes runs where the gap is large by chance rather than by structure; the p-value guard catches most but not all of these cases.

## Future Directions

The extension to linear regression is straightforward.