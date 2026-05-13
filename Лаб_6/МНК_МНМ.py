import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pandas as pd

x = np.arange(-1.8, 2.1, 0.2)   # 20 точек
n = len(x)
a_true, b_true = 2.0, 2.0

np.random.seed(42)
epsilon = np.random.normal(0, 1, n)
y_clean = b_true + a_true * x + epsilon

y_outliers = y_clean.copy()
y_outliers[0] += 10
y_outliers[-1] -= 10

def ols_fit(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    a = (np.mean(x * y) - x_mean * y_mean) / (np.mean(x ** 2) - x_mean ** 2)
    b = y_mean - a * x_mean
    return a, b

def lad_loss(params, x, y):
    a, b = params
    return np.sum(np.abs(y - (a * x + b)))

def lad_fit(x, y):
    # начальное приближение – МНК
    a0, b0 = ols_fit(x, y)
    res = minimize(lad_loss, x0=[a0, b0], args=(x, y), method='BFGS')
    return res.x[0], res.x[1]

def get_stats(a_hat, b_hat):
    da = abs(a_true - a_hat)
    db = abs(b_true - b_hat)
    sa = (da / a_true) * 100
    sb = (db / b_true) * 100
    return [a_hat, da, sa, b_hat, db, sb]

def print_results_table(x_data, y_data, title):
    a_ols, b_ols = ols_fit(x_data, y_data)
    a_lad, b_lad = lad_fit(x_data, y_data)

    columns = ["Метод", "a", "Δ a", "δ a, %", "b", "Δ b", "δ b, %"]
    data = [
        ["МНК"] + get_stats(a_ols, b_ols),
        ["МНМ"] + get_stats(a_lad, b_lad)
    ]
    df = pd.DataFrame(data, columns=columns)
    print(f"\n{title}")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return (a_ols, b_ols), (a_lad, b_lad)

def plot_lab(x, y, params_ols, params_lad, title):
    plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle=':', alpha=0.6)

    # Точки данных
    plt.scatter(x, y, color='cornflowerblue', alpha=0.7, edgecolors='darkblue', label='Данные')
    # Истинная прямая
    plt.plot(x, b_true + a_true * x, 'k--', linewidth=1.5, alpha=0.7, label='Истинный закон')
    # МНК
    plt.plot(x, params_ols[1] + params_ols[0] * x, color='seagreen', lw=2.5, label='МНК (OLS)')
    # МНМ
    plt.plot(x, params_lad[1] + params_lad[0] * x, color='orange', lw=2.5, label='МНМ (LAD)')

    plt.title(title, fontsize=12)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Без выбросов
    p_ols_1, p_lad_1 = print_results_table(x, y_clean, "Без выбросов")
    plot_lab(x, y_clean, p_ols_1, p_lad_1, "Линейная регрессия (без выбросов)")

    # С выбросами
    p_ols_2, p_lad_2 = print_results_table(x, y_outliers, "С выбросами (y1+10, y20-10)")
    plot_lab(x, y_outliers, p_ols_2, p_lad_2, "Линейная регрессия (с выбросами)")
