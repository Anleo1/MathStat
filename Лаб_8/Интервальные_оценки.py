import numpy as np
import scipy.stats as stats
import pandas as pd

np.random.seed(42)

# Генерация выборок
n1, n2 = 20, 100
sample1 = np.random.normal(0, 1, n1)
sample2 = np.random.normal(0, 1, n2)

def ci_mean(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = np.std(data, ddof=1) / np.sqrt(n)
    return stats.t.interval(confidence, df=n-1, loc=mean, scale=sem)

def ci_var(data, confidence=0.95):
    n = len(data)
    var = np.var(data, ddof=1)
    chi2_low = stats.chi2.ppf((1-confidence)/2, n-1)
    chi2_high = stats.chi2.ppf(1-(1-confidence)/2, n-1)
    return ((n-1)*var / chi2_high, (n-1)*var / chi2_low)

mean1 = np.mean(sample1)
var1 = np.var(sample1, ddof=1)
ci_mean_1 = ci_mean(sample1)
ci_var_1 = ci_var(sample1)

mean2 = np.mean(sample2)
var2 = np.var(sample2, ddof=1)
ci_mean_2 = ci_mean(sample2)
ci_var_2 = ci_var(sample2)

if var1 > var2:
    F = var1 / var2
    df_num, df_den = n1-1, n2-1
else:
    F = var2 / var1
    df_num, df_den = n2-1, n1-1
F_crit = stats.f.ppf(0.975, df_num, df_den)
p_value = 2 * min(stats.f.cdf(F, df_num, df_den), 1 - stats.f.cdf(F, df_num, df_den))
reject = F > F_crit

df1 = pd.DataFrame({
    'Выборка': ['n=20', 'n=100'],
    'Среднее': [round(mean1, 2), round(mean2, 2)],
    'Дисперсия': [round(var1, 2), round(var2, 2)],
    'ДИ для среднего (95%)': [f"[{ci_mean_1[0]:.2f}; {ci_mean_1[1]:.2f}]", f"[{ci_mean_2[0]:.2f}; {ci_mean_2[1]:.2f}]"],
    'ДИ для дисперсии (95%)': [f"[{ci_var_1[0]:.2f}; {ci_var_1[1]:.2f}]", f"[{ci_var_2[0]:.2f}; {ci_var_2[1]:.2f}]"]
})

# Таблица 2: Результаты F-теста
df2 = pd.DataFrame({
    'Параметр': ['F-статистика', 'Критическое значение (0.975)', 'p-value'],
    'Значение': [round(F, 2), round(F_crit, 2), round(p_value, 3)]
})

print("    Доверительные интервалы   ")
print(df1.to_string(index=False))
print("\n    F-тест для равенства дисперсий    ")
print(df2.to_string(index=False))
