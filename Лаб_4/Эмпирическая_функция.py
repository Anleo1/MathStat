import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform
from scipy.stats import gaussian_kde

np.random.seed(42)
n_samples = [20, 60, 100]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

def get_optimal_bins(sample, n):
    #Правило Фридмана-Диакониса (Стерджес для n<=10)
    if n <= 10:
        return int(np.ceil(np.log2(n) + 1))
    q75, q25 = np.percentile(sample, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return int(np.sqrt(n))
    bin_width = 2 * iqr / (n ** (1/3))
    k = int(np.ceil((sample.max() - sample.min()) / bin_width))
    return max(5, min(k, 45))

def merge_empty_bins(bins, counts):
    if np.all(counts > 0):
        return bins, counts
    non_empty = counts > 0
    if not np.any(non_empty):
        return np.array([bins[0], bins[-1]]), np.array([0])
    first = np.where(non_empty)[0][0]
    last = np.where(non_empty)[0][-1]
    new_bins = [bins[first]]
    new_counts = []
    i = first
    while i <= last:
        if counts[i] > 0:
            new_bins.append(bins[i+1])
            new_counts.append(counts[i])
            i += 1
        else:
            if new_counts:
                new_bins[-1] = bins[i+1]
            i += 1
    return np.array(new_bins), np.array(new_counts)

distributions = {
    "Normal": {
        "dist": norm(0, 1),
        "x_range": (-4, 4),
        "is_discrete": False,
        "theoretical_pdf": lambda x: norm.pdf(x, 0, 1),
        "theoretical_cdf": lambda x: norm.cdf(x, 0, 1)
    },
    "Cauchy": {
        "dist": cauchy(0, 1),
        "x_range": (-4, 4),
        "is_discrete": False,
        "theoretical_pdf": lambda x: cauchy.pdf(x, 0, 1),
        "theoretical_cdf": lambda x: cauchy.cdf(x, 0, 1)
    },
    "Laplace": {
        "dist": laplace(0, 1 / np.sqrt(2)),
        "x_range": (-4, 4),
        "is_discrete": False,
        "theoretical_pdf": lambda x: laplace.pdf(x, 0, 1 / np.sqrt(2)),
        "theoretical_cdf": lambda x: laplace.cdf(x, 0, 1 / np.sqrt(2))
    },
    "Poisson": {
        "dist": poisson(10),
        "x_range": (6, 14),
        "is_discrete": True,
        "theoretical_pmf": lambda k: poisson.pmf(k, 10),
        "theoretical_cdf": lambda x: poisson.cdf(x, 10)
    },
    "Uniform": {
        "dist": uniform(-np.sqrt(3), 2 * np.sqrt(3)),
        "x_range": (-4, 4),
        "is_discrete": False,
        "theoretical_pdf": lambda x: uniform.pdf(x, -np.sqrt(3), 2 * np.sqrt(3)),
        "theoretical_cdf": lambda x: uniform.cdf(x, -np.sqrt(3), 2 * np.sqrt(3))
    }
}

def ecdf_full_range(sample, x_min, x_max):
    x_sorted = np.sort(sample)
    y_sorted = np.arange(1, len(sample) + 1) / len(sample)
    x_full = np.concatenate(([x_min], x_sorted, [x_max]))
    y_full = np.concatenate(([0], y_sorted, [1]))
    return x_full, y_full

def plot_ecdf_and_kde(dist_name, dist_info, n_samples_list):
    dist = dist_info["dist"]
    x_range = dist_info["x_range"]
    is_discrete = dist_info["is_discrete"]
    x_min, x_max = x_range

    fig, axes = plt.subplots(2, len(n_samples_list), figsize=(15, 10))
    fig.suptitle(f'Распределение {dist_name}', fontsize=16, fontweight='bold')

    for i, n in enumerate(n_samples_list):
        sample = dist.rvs(size=n)

        ax1 = axes[0, i]
        x_ecdf, y_ecdf = ecdf_full_range(sample, x_min, x_max)
        ax1.step(x_ecdf, y_ecdf, where='post', label='Эмпирическая ФР',
                 color=colors[i], linewidth=2)
        x_theor = np.linspace(x_min, x_max, 1000)
        if is_discrete:
            y_theor_cdf = [dist_info["theoretical_cdf"](xx) for xx in x_theor]
        else:
            y_theor_cdf = dist_info["theoretical_cdf"](x_theor)
        ax1.plot(x_theor, y_theor_cdf, '--', color='black',
                 label='Теоретическая ФР', linewidth=2)
        ax1.set_title(f'n = {n}')
        ax1.set_xlabel('x')
        ax1.set_ylabel('F(x)')
        ax1.legend(loc='lower right', fontsize=8, framealpha=0.5)
        ax1.grid(alpha=0.3, linestyle='--')
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(-0.05, 1.05)

        ax2 = axes[1, i]

        # KDE
        kde = gaussian_kde(sample)
        x_kde = np.linspace(x_min, x_max, 500)
        y_kde = kde.evaluate(x_kde)
        ax2.plot(x_kde, y_kde, '-', label='Ядерная оценка', linewidth=2, color=colors[i])

        # Теоретическая кривая
        if is_discrete:
            k_vals = np.arange(int(x_min), int(x_max) + 1)
            p_vals = [dist_info["theoretical_pmf"](k) for k in k_vals]
            ax2.plot(k_vals, p_vals, '--', color='black', linewidth=2, label='Теория')
            ax2.plot(k_vals, p_vals, 'o', color='black', markersize=6)
        else:
            y_theor_pdf = dist_info["theoretical_pdf"](x_kde)
            ax2.plot(x_kde, y_theor_pdf, '--', color='black', label='Теория', linewidth=2)

        # Гистограмма с объединением пустых бинов
        if is_discrete:
            bins_initial = np.arange(int(x_min) - 0.5, int(x_max) + 1.5, 1)
        else:
            k_opt = get_optimal_bins(sample, n)
            bins_initial = np.linspace(x_min, x_max, k_opt + 1)
        counts, bins = np.histogram(sample, bins=bins_initial)
        merged_bins, _ = merge_empty_bins(bins, counts)
        ax2.hist(sample, bins=merged_bins, density=True, alpha=0.3,
                 color='gray', label='Гистограмма')
        if is_discrete:
            ax2.set_xticks(k_vals)
            ax2.set_xticklabels(k_vals.astype(int))

        ax2.set_title(f'n = {n}')
        ax2.set_xlabel('x')
        ax2.set_ylabel('Плотность / Вероятность')
        ax2.legend(loc='upper left', fontsize=8, framealpha=0.5)
        ax2.grid(alpha=0.3, linestyle='--')
        ax2.set_xlim(x_min, x_max)

    plt.tight_layout()
    plt.savefig(f'{dist_name}.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Сохранён: {dist_name}.png")

# Запуск
for dist_name, dist_info in distributions.items():
    plot_ecdf_and_kde(dist_name, dist_info, n_samples)
