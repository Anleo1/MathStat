import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, cauchy, laplace, poisson, uniform

n_samples = [20, 100]
n_experiments = 1000
np.random.seed(42)

distributions = {
    "Normal": norm(0, 1),
    "Cauchy": cauchy(0, 1),
    "Laplace": laplace(0, 1 / np.sqrt(2)),
    "Poisson": poisson(10),
    "Uniform": uniform(-np.sqrt(3), 2 * np.sqrt(3))
}

colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2']

def count_outliers(sample):
    #Подсчёт выбросов
    Q1 = np.percentile(sample, 25)
    Q3 = np.percentile(sample, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = sample[(sample < lower_bound) | (sample > upper_bound)]
    return len(outliers)

def plot_boxplot(n, title, filename):
    #Построение отдельного боксплота для всех распределений при заданном n
    plt.figure(figsize=(10, 6))

    data = []
    tick_labels = []
    for name, dist in distributions.items():
        sample = dist.rvs(size=n)
        data.append(sample)
        tick_labels.append(name)

    bp = plt.boxplot(data, tick_labels=tick_labels, patch_artist=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)

    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)

    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1)

    for cap in bp['caps']:
        cap.set_color('black')
        cap.set_linewidth(1)

    for flier in bp['fliers']:
        flier.set_marker('o')
        flier.set_markerfacecolor('black')
        flier.set_markeredgecolor('black')
        flier.set_markersize(4)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('Значения', fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Сохранён: {filename}")


def calculate_outlier_fraction(dist, n, n_experiments):
    #Вычисление средней доли выбросов для заданного распределения и n
    total_outliers = 0
    total_elements = n * n_experiments

    for _ in range(n_experiments):
        sample = dist.rvs(size=n)
        total_outliers += count_outliers(sample)

    return total_outliers / total_elements * 100  # в процентах


print("\n1. Построение боксплотов")
plot_boxplot(20, 'Боксплоты Тьюки для n = 20', 'boxplot_n20.png')
plot_boxplot(100, 'Боксплоты Тьюки для n = 100', 'boxplot_n100.png')

# Расчёт средней доли выбросов (1000 экспериментов)
print("\n2. Расчёт средней доли выбросов (1000 экспериментов):")

results = {}

for dist_name, dist in distributions.items():
    print(f"\nОбработка: {dist_name}")
    fractions = []
    for n in n_samples:
        fraction = calculate_outlier_fraction(dist, n, n_experiments)
        fractions.append(fraction)
        print(f"  n = {n}: {fraction:.4f}% (всего элементов: {n * n_experiments})")
    results[dist_name] = fractions


print(f"\n{'Распределение':<12} {'n=20':>12} {'n=100':>12}")
for dist_name, fractions in results.items():
    print(f"{dist_name:<12} {fractions[0]:>11.2f}% {fractions[1]:>11.2f}%")

# Сохранение
import csv

with open("outliers_results.csv", "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Распределение", "n=20 (%)", "n=100 (%)"])
    for dist_name, fractions in results.items():
        writer.writerow([dist_name, f"{fractions[0]:.2f}", f"{fractions[1]:.2f}"])

print("\n5. Результаты сохранены в outliers_results.csv")
