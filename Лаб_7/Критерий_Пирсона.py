import numpy as np
import scipy.stats as stats
import pandas as pd


def chi2_table(sample, dist_name, n_val):
    n = len(sample)
    mu_hat = np.mean(sample)
    sigma_hat = np.std(sample, ddof=0)

    k = max(3, int(np.sqrt(n)))
    while k > 2 and n / k < 5:
        k -= 1

    prob = 1.0 / k
    quantiles = [stats.norm.ppf(i * prob, loc=mu_hat, scale=sigma_hat) for i in range(1, k)]
    bins = [-np.inf] + quantiles + [np.inf]
    observed, _ = np.histogram(sample, bins=bins)
    expected = np.full(k, n / k)
    chi2_stat = np.sum((observed - expected) ** 2 / expected)

    return pd.Series({
        'Распределение': dist_name,
        'n': n_val,
        'mu': round(mu_hat, 2),
        'sigma': round(sigma_hat, 2),
        'k': k,
        'Кси2': round(chi2_stat, 2)
    })


np.random.seed(42)

results = []
# Normal
sample_norm = np.random.normal(0, 1, 100)
results.append(chi2_table(sample_norm, "Нормальное", 100))
# Uniform
sample_unif_20 = np.random.uniform(-np.sqrt(3), np.sqrt(3), 20)
results.append(chi2_table(sample_unif_20, "Равномерное", 20))
sample_unif_300 = np.random.uniform(-np.sqrt(3), np.sqrt(3), 1000)
results.append(chi2_table(sample_unif_300, "Равномерное", 1000))
# Laplace
sample_laplace_20 = np.random.laplace(0, 1 / np.sqrt(2), 20)
results.append(chi2_table(sample_laplace_20, "Лапласа", 20))
sample_laplace_300 = np.random.laplace(0, 1 / np.sqrt(2), 1000)
results.append(chi2_table(sample_laplace_300, "Лапласа", 1000))

df = pd.DataFrame(results)
print(df.to_string(index=False))
