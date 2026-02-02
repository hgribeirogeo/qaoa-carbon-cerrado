import numpy as np
import pandas as pd
from scipy import stats
import os

# ==============================================================================
# 1. DADOS EXPERIMENTAIS CONSOLIDADOS (n=7)
# ==============================================================================
# Scores individuais obtidos via QAOA + ZNE (Quadrático)
zne_scores = np.array([45.42, 52.81, 52.81, 69.64, 58.68, 58.72, 63.05])
greedy_baseline = 44.42

def calculate_cohens_d(sample, pop_mean):
    """Calcula o Cohen's d para uma amostra contra uma média populacional."""
    return (np.mean(sample) - pop_mean) / np.std(sample, ddof=1)

def perform_bootstrap(data, n_iterations=1000):
    """Gera intervalo de confiança de 95% via bootstrap não-paramétrico."""
    means = []
    for _ in range(n_iterations):
        sample = np.random.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))
    return np.percentile(means, [2.5, 97.5])

# ==============================================================================
# 2. PROCESSAMENTO ESTATÍSTICO
# ==============================================================================
mean_score = np.mean(zne_scores)
std_dev = np.std(zne_scores, ddof=1)
cv = (std_dev / mean_score) * 100

# Teste t de uma amostra (comparando n=7 contra o baseline fixo)
t_stat, p_value = stats.ttest_1samp(zne_scores, greedy_baseline)
d_effect = calculate_cohens_d(zne_scores, greedy_baseline)
ci_lower, ci_upper = perform_bootstrap(zne_scores)

# ==============================================================================
# 3. RELATÓRIO FINAL
# ==============================================================================
print("-" * 50)
print("STATISTICAL ANALYSIS REPORT (QAOA + ZNE vs GREEDY)")
print("-" * 50)
print(f"Sample Size (n):      7 independent runs")
print(f"Mean Score:           {mean_score:.2f}")
print(f"Standard Deviation:   {std_dev:.2f}")
print(f"Coeff. Variation:     {cv:.2f}%")
print(f"Greedy Baseline:      {greedy_baseline:.2f}")
print("-" * 50)
print(f"T-statistic:          {t_stat:.4f}")
print(f"P-value:              {p_value:.4f}")
print(f"Cohen's d:            {d_effect:.2f} (Large Effect)")
print(f"95% CI (Bootstrap):   [{ci_lower:.2f}, {ci_upper:.2f}]")
print("-" * 50)

# Verificação de Superioridade
if p_value < 0.05:
    improvement = ((mean_score / greedy_baseline) - 1) * 100
    print(f"RESULT: Statistically Significant Superiority (+{improvement:.1f}%)")
else:
    print("RESULT: No significant difference detected.")