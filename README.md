# NaturalImputation

**NaturalImputation** is a simple but effective target-driven imputation method for Logistic Regression. For a given feature $x$ and target $y$, let $\beta_0$ and $\beta_1$ be the coefficients that minimize cross-entropy for the relationship:

$$\ln\left(\frac{y'}{1-y'}\right) = \beta_0 + \beta_1 x'$$

Where $x'$ and $y'$ represent the subsets of $x$ and $y$ such that $x$ is non-missing. Then, $x$ is **naturally imputed** at the value:

$$x^{\ast} = \frac{\ln\left(\frac{y^{\ast}}{1-y^{\ast}}\right) - \beta_0}{\beta_1}$$

where $y^{\ast}$ represents the mean of the target variable for the missing observations.

This method avoids both the naïveté of mean/median imputation, and the complex dependence on other features inherent in more robust methods like MICE. It is effective when 1) $x'$ sufficiently predicts $y'$ and 2) $y^*$ is bounded away from $\bar{y}$. In the failure of either of these conditions, particularly 1), mean or median imputation can outperform NaturalImputation.

## Example
```
>>> import numpy as np
>>> import pandas as pd
>>> from naturalimputation import impute_logistic
>>> from sklearn.datasets import make_classification
>>> 
>>> rng = np.random.default_rng(0)
>>> X, y = make_classification(
...     random_state=rng.integers(0, 2**32 - 1)
... )
>>> X, y = pd.DataFrame(X), pd.Series(y)
>>> test = rng.random(len(X)) < 0.25
>>>
>>> impute_logistic(X[0], y, test)[:5]
0    0.259723
1   -1.231660
2    1.154356
3    0.464447
4   -0.106562
dtype: float64
```

<!-- ## Future Example to illustrate when conditions support naturalimputation
import numpy as np
import pandas as pd
from demo import run_experiment
from scipy import stats

results = {}

for steepness in [1.05, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 3, 4, 5]:

    mean_imp_results, logistic_imp_results = run_experiment(
        steepness=steepness,
        n_iterations=250,
        n_samples=10_000,
        flip_y=0.4,
        random_state=0,
    )

    results[steepness] = {
        'mean_difference': np.mean(logistic_imp_results - mean_imp_results),
        'p_value': stats.ttest_rel(logistic_imp_results, mean_imp_results)[1]
    }

result_df = pd.DataFrame(results).T.round(2)
result_df.index.name = 'steepness'
print(result_df) -->