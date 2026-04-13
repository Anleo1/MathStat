import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform

# Параметры распределений
distributions = {
    "Normal": norm(0, 1),
    "Cauchy": cauchy(0, 1),
    "Laplace": laplace(0, 1/np.sqrt(2)),
    "Poisson": poisson(5),
    "Uniform": uniform(-np.sqrt(3), 2*np.sqrt(3))
}

n_samples = [10, 50, 1000]
colors = ["blue", "green", "red"]

fig, axes = plt.subplots(len(distributions), len(n_samples), figsize=(15, 12))

for i, (name, dist) in enumerate(distributions.items()):
    for j, n in enumerate(n_samples):
        ax = axes[i, j]
        # Генерация выборки
        if name == "Poisson":
            sample = dist.rvs(size=n)
            bins = np.arange(0, max(sample)+2) - 0.5
        else:
            sample = dist.rvs(size=n)
            # Правило Стерджеса
            k = int(np.ceil(np.log2(n) + 1))
            bins = k

        # Гистограмма
        ax.hist(sample, bins=bins, density=True, alpha=0.6, color=colors[j], label=f"n={n}")

        # Теоретическая плотность
        if name == "Poisson":
            x_vals = np.arange(0, max(sample)+1)
            pdf_vals = dist.pmf(x_vals)
            ax.plot(x_vals, pdf_vals, 'ro-', label="Теоретическая PMF")
        else:
            x_vals = np.linspace(dist.ppf(0.001), dist.ppf(0.999), 200)
            pdf_vals = dist.pdf(x_vals)
            ax.plot(x_vals, pdf_vals, 'k-', linewidth=2, label="Теоретическая плотность")

        ax.set_title(f"{name}, n={n}")
        ax.legend()
        ax.grid(True)

plt.tight_layout()
plt.show()