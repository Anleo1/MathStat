import numpy as np
from scipy.stats import norm, cauchy, laplace, poisson, uniform

def sample_mean(x): return np.mean(x)

def sample_median(x): return np.median(x)

def z_R(x): return (np.min(x) + np.max(x)) / 2

def z_Q(x): return (np.percentile(x, 25) + np.percentile(x, 75)) / 2

def z_tr(x, trim=0.1):
    n = len(x)
    r = int(n * trim)
    return np.mean(np.sort(x)[r:n - r])

def calc_statistics(values):
    return np.mean(values), np.var(values, ddof=0)

n_samples = [10, 100, 1000]
n_experiments = 1000
np.random.seed(42)

distributions = {
    "Normal": norm(0, 1),
    "Cauchy": cauchy(0, 1),
    "Laplace": laplace(0, 1 / np.sqrt(2)),
    "Poisson": poisson(10),
    "Uniform": uniform(-np.sqrt(3), 2 * np.sqrt(3))
}

char_names = ["mean", "median", "zR", "zQ", "ztr"]
char_labels = ["x̄", "med", "z_R", "z_Q", "z_tr"]

for dist_name, dist in distributions.items():
    with open(f"{dist_name}.csv", "w", encoding="utf-8-sig") as f:
        # Заголовки
        f.write("n;characteristic;E(z);D(z);sqrt(D(z));E(z) ± sqrt(D(z))\n")

        for n in n_samples:
            results = {key: [] for key in char_names}

            for _ in range(n_experiments):
                sample = dist.rvs(size=n)
                results["mean"].append(sample_mean(sample))
                results["median"].append(sample_median(sample))
                results["zR"].append(z_R(sample))
                results["zQ"].append(z_Q(sample))
                results["ztr"].append(z_tr(sample))

            for key, label in zip(char_names, char_labels):
                mean_val, var_val = calc_statistics(results[key])
                std_val = np.sqrt(var_val)
                f.write(f"{n};{label};{mean_val:.5f};{var_val:.5f};{std_val:.5f};{mean_val:.5f} ± {std_val:.5f}\n")

    print(f"Сохранён: {dist_name}.csv")

print("\nГотово! Созданы CSV файлы :")
print("Normal.csv, Cauchy.csv, Laplace.csv, Poisson.csv, Uniform.csv")
