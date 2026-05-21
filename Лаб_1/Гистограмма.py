import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform

def get_optimal_bins(sample, n, dist_name):
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
    #Объединяет пустые бины с соседними непустыми
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

def draw_histogram(ax, n, dist, dist_name, x_range, is_discrete=False):
    sample = dist.rvs(size=n)

    # Для Коши выводим информацию о хвостах
    if dist_name.startswith("Коши"):
        ax.set_xlim(-10, 10)
        left_tail = np.sum(sample < x_range[0])
        right_tail = np.sum(sample > x_range[1])

        print(f"\nКоши, n={n}")
        print(f"Минимум: {sample.min():.3f}")
        print(f"Максимум: {sample.max():.3f}")
        print(f"Левый хвост (< {x_range[0]}): {left_tail}")
        print(f"Правый хвост (> {x_range[1]}): {right_tail}")

        # Расширяем диапазон отображения
        q01, q99 = np.percentile(sample, [1, 99])
        x_range = (
            min(x_range[0], q01),
            max(x_range[1], q99)
        )

    # Для всех распределений одинаково: определяем k по правилу, строим бины,
    # объединяем пустые, рисуем гистограмму.
    k = get_optimal_bins(sample, n, dist_name)
    bins = np.linspace(x_range[0], x_range[1], k + 1)
    counts, _ = np.histogram(sample, bins=bins)
    merged_bins, _ = merge_empty_bins(bins, counts)

    # Гистограмма
    ax.hist(sample, bins=merged_bins, density=True, alpha=0.7,
            color='green', edgecolor='green', linewidth=0,
            range=(merged_bins[0], merged_bins[-1]))

    # Теоретическая кривая
    if is_discrete:
        # Для Пуассона — точки (PMF)
        x_vals = np.arange(int(x_range[0]), int(x_range[1]) + 1)
        y_vals = dist.pmf(x_vals)
        ax.plot(x_vals, y_vals, 'ro-', linewidth=2, markersize=4, label='Теория')
        # Целочисленные метки на оси X
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_vals.astype(int))
        ax.set_ylabel('Вероятность')
    else:
        # Для непрерывных — плотность
        x_vals = np.linspace(x_range[0], x_range[1], 500)
        y_vals = dist.pdf(x_vals)
        ax.plot(x_vals, y_vals, 'r-', linewidth=2.5, label='Теория')
        ax.set_ylabel('Плотность')

    method = "Фридман-Диаконис" if n > 10 else "Стерджес"
    k_fact = len(merged_bins) - 1
    ax.set_title(f'n = {n}\n{method}, k={k_fact}', fontsize=10)
    ax.set_xlabel('x', fontsize=10)
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

def main():
    n_values = [10, 100, 1000]
    distributions = {
        "Нормальное N(0,1)": (norm(0,1), (-5,5), False),
        "Коши C(0,1)": (cauchy(0,1), (-5,5), False),
        "Лапласа L(0,1/√2)": (laplace(0,1/np.sqrt(2)), (-5,5), False),
        "Пуассон P(10)": (poisson(10), (0,20), True),
        "Равномерное U(-√3,√3)": (uniform(-np.sqrt(3),2*np.sqrt(3)), (-5,5), False),
    }
    for dist_name, (dist, x_range, is_discrete) in distributions.items():
        fig, axes = plt.subplots(1, 3, figsize=(15,5))
        fig.suptitle(f'Распределение: {dist_name}', fontsize=16, fontweight='bold')
        for i, n in enumerate(n_values):
            draw_histogram(axes[i], n, dist, dist_name, x_range, is_discrete)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
