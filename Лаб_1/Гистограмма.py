import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform


def get_optimal_bins(sample, n, dist_name):

    # Для очень маленьких выборок (n=10) используем правило Стерджеса
    if n <= 10:
        k = int(np.ceil(np.log2(n) + 1))
        return k

    # Для остальных (n=100, 1000) используем правило Фридмана-Диакониса
    q75, q25 = np.percentile(sample, [75, 25])
    iqr = q75 - q25  # межквартильный размах

    # Защита от нулевого IQR (если все числа одинаковые)
    if iqr == 0:
        return int(np.sqrt(n))

    # Формула Фридмана-Диакониса
    bin_width = 2 * iqr / (n ** (1 / 3))
    k = int(np.ceil((sample.max() - sample.min()) / bin_width))

    # Ограничиваем число бинов
    return max(5, min(k, 45))

def draw_histogram(ax, n, dist, dist_name, x_range, is_discrete=False):
    sample = dist.rvs(size=n)

    if is_discrete:
        max_val = max(sample) if len(sample) > 0 else 20
        bins = np.arange(-0.5, max(max_val + 2, 21), 1)
        method_name = "Целочисленные"
        k_display = len(bins) - 1
    else:
        k = get_optimal_bins(sample, n, dist_name)
        bins = k
        method_name = "Фридман-Диаконис" if n > 10 else "Стерджес"
        k_display = k

    # Рисуем гистограмму
    if is_discrete:
        ax.hist(sample, bins=bins, density=True, alpha=0.7,
                color='green', edgecolor='green', linewidth=0.5)
        ax.set_xlim(x_range[0], x_range[1])
    else:
        ax.hist(sample, bins=bins, density=True, alpha=0.7,
                color='green', edgecolor='green', linewidth=0.5,
                range=x_range)

    # Рисуем теоретическую кривую
    if is_discrete:
        # Для Пуассона — точки (функция вероятности)
        x_vals = np.arange(0, x_range[1] + 1)
        y_vals = dist.pmf(x_vals)
        ax.plot(x_vals, y_vals, 'ro-', linewidth=2, markersize=4, label='Теория')
    else:
        # Для непрерывных — плавная линия (плотность)
        x_vals = np.linspace(x_range[0], x_range[1], 500)
        y_vals = dist.pdf(x_vals)
        ax.plot(x_vals, y_vals, 'r-', linewidth=2.5, label='Теория')

    ax.set_title(f'n = {n}\n{method_name}, k={k_display}', fontsize=10)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('Плотность', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

def main():
    n_values = [10, 100, 1000]

    distributions = {
        "Нормальное N(0,1)": {
            "dist": norm(0, 1),
            "x_range": (-5, 5),
            "is_discrete": False
        },
        "Коши C(0,1)": {
            "dist": cauchy(0, 1),
            "x_range": (-5, 5),
            "is_discrete": False
        },
        "Лапласа L(0, 1/√2)": {
            "dist": laplace(0, 1 / np.sqrt(2)),
            "x_range": (-5, 5),
            "is_discrete": False
        },
        "Пуассон P(10)": {
            "dist": poisson(10),
            "x_range": (0, 20),
            "is_discrete": True
        },
        "Равномерное U(-√3, √3)": {
            "dist": uniform(-np.sqrt(3), 2 * np.sqrt(3)),
            "x_range": (-5, 5),
            "is_discrete": False
        }
    }

    # Строим графики для каждого распределения
    for dist_name, info in distributions.items():

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Распределение: {dist_name}', fontsize=16, fontweight='bold')

        # Рисуем гистограммы для n=10, 100, 1000
        for i, n in enumerate(n_values):
            draw_histogram(
                ax=axes[i],
                n=n,
                dist=info["dist"],
                dist_name=dist_name,
                x_range=info["x_range"],
                is_discrete=info["is_discrete"]
            )

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
