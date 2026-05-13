import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, chi2
from matplotlib.patches import Ellipse

np.random.seed(42)

def pearson_corr(x, y):
    return np.corrcoef(x, y)[0, 1]

def spearman_corr(x, y):
    return spearmanr(x, y)[0]

def quadrant_corr(x, y):
    mx, my = np.median(x), np.median(y)
    x_sign = np.sign(x - mx)
    y_sign = np.sign(y - my)
    prod = x_sign * y_sign
    n_plus = np.sum(prod == 1)
    n_minus = np.sum(prod == -1)
    return (n_plus - n_minus) / len(x)

def gen_normal(rho, n):
    cov = [[1, rho], [rho, 1]]
    data = np.random.multivariate_normal([0, 0], cov, n)
    return data[:, 0], data[:, 1]

def gen_mixture(n):
    n1 = int(0.9 * n)
    n2 = n - n1
    cov1 = [[1, 0.9], [0.9, 1]]
    cov2 = [[10, -0.9 * 10], [-0.9 * 10, 10]]
    data1 = np.random.multivariate_normal([0, 0], cov1, n1)
    data2 = np.random.multivariate_normal([0, 0], cov2, n2)
    data = np.vstack([data1, data2])
    np.random.shuffle(data)
    return data[:, 0], data[:, 1]

def draw_ellipse(x, y, ax, level=0.95):
    cov = np.cov(x, y)
    mean = np.mean(x), np.mean(y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    chi2_val = chi2.ppf(level, 2)
    width, height = 2 * np.sqrt(vals * chi2_val)
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=theta,
                      edgecolor='red', fc='none', lw=2)
    ax.add_patch(ellipse)

def plot_for_n(gen_func, title, n_values=[20,60,100], savefile=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14)
    for i, n in enumerate(n_values):
        x, y = gen_func(n)
        axes[i].scatter(x, y, alpha=0.5, s=10)
        draw_ellipse(x, y, axes[i])
        r_p = pearson_corr(x, y)
        r_s = spearman_corr(x, y)
        r_q = quadrant_corr(x, y)
        axes[i].set_title(f'n = {n}\nПирсон={r_p:.2f}, Спирмен={r_s:.2f}, Квадр={r_q:.2f}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].grid(alpha=0.3)
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"График сохранён: {savefile}")
    plt.show()

def simulate_stats(gen_func, n_values, n_rep=1000):
    results = {}
    for n in n_values:
        pearson_vals = []
        spearman_vals = []
        quadrant_vals = []
        for _ in range(n_rep):
            x, y = gen_func(n)
            pearson_vals.append(pearson_corr(x, y))
            spearman_vals.append(spearman_corr(x, y))
            quadrant_vals.append(quadrant_corr(x, y))
        results[n] = {
            'pearson': (np.mean(pearson_vals), np.var(pearson_vals)),
            'spearman': (np.mean(spearman_vals), np.var(spearman_vals)),
            'quadrant': (np.mean(quadrant_vals), np.var(quadrant_vals))
        }
    return results

def print_table(stats, title):
    print(f"\n{title}")
    print(f"{'n':<6} {'Коэффициент':<12} {'E(z)':<12} {'D(z)':<12}")
    print("-" * 45)
    for n in sorted(stats.keys()):
        print(f"{n:<6} Пирсон      {stats[n]['pearson'][0]:<12.4f} {stats[n]['pearson'][1]:<12.5f}")
        print(f"{n:<6} Спирмен     {stats[n]['spearman'][0]:<12.4f} {stats[n]['spearman'][1]:<12.5f}")
        print(f"{n:<6} Квадрантный {stats[n]['quadrant'][0]:<12.4f} {stats[n]['quadrant'][1]:<12.5f}")
        if n != list(stats.keys())[-1]:
            print()

def main():
    n_values = [20, 60, 100]
    n_rep = 1000
    rhos = [0, 0.5, 0.9]

    for rho in rhos:
        gen = lambda n: gen_normal(rho, n)
        stats = simulate_stats(gen, n_values, n_rep)
        print_table(stats, f"Двумерное нормальное распределение, ρ = {rho}")

        fname = f"normal_rho_{rho}.png"
        plot_for_n(gen, f"Нормальное распределение, ρ = {rho}", n_values, savefile=fname)

    stats_mix = simulate_stats(gen_mixture, n_values, n_rep)
    print_table(stats_mix, "Смесь: 0.9N(0,0,1,1,0.9) + 0.1N(0,0,10,10,-0.9)")
    plot_for_n(gen_mixture, "Смесь распределений", n_values, savefile="mixture.png")

if __name__ == "__main__":
    main()
