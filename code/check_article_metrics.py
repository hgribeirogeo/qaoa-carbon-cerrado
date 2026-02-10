"""
Sanity-check script for manuscript metrics.

This script reproduces the primary statistical metrics reported in:
"QAOA with Zero Noise Extrapolation for Carbon Credit Portfolio Optimization"

Data source:
- resultados_consolidados_v7.json

Purpose:
- Quick verification of reported mean, SD, effect size, p-value, and CI.
- Not used to generate figures or primary results.

Changing this script MUST NOT change manuscript conclusions.
"""

import numpy as np
from scipy import stats

# ==============================================================================
# 1. DADOS EXPERIMENTAIS CONSOLIDADOS (n=7) — alinhado ao resultados_consolidados_v7.json
# ==============================================================================
# QAOA + ZNE (Quadrático) por run:
# Run1=58.69, Run2=52.70, Run3=47.84, Run4=69.64, Run5=58.68, Run6=58.72, Run7=63.05
zne_scores = np.array([58.69, 52.70, 47.84, 69.64, 58.68, 58.72, 63.05], dtype=float)

greedy_baseline = 44.42

RANDOM_SEED = 42
BOOTSTRAP_B = 1000   

def cohens_d_onesample(sample, pop_mean):
    """Cohen's d para teste 1-amostra (diferença padronizada)."""
    return (np.mean(sample) - pop_mean) / np.std(sample, ddof=1)

def bootstrap_ci_mean(data, n_iterations=BOOTSTRAP_B, alpha=0.05, seed=RANDOM_SEED):
    """IC bootstrap (percentil) para a média."""
    rng = np.random.default_rng(seed)
    means = np.empty(n_iterations, dtype=float)
    n = len(data)
    for i in range(n_iterations):
        sample = rng.choice(data, size=n, replace=True)
        means[i] = np.mean(sample)
    lo = np.percentile(means, 100 * (alpha/2))
    hi = np.percentile(means, 100 * (1 - alpha/2))
    return lo, hi

# ==============================================================================
# 2. PROCESSAMENTO ESTATÍSTICO
# ==============================================================================
mean_score = np.mean(zne_scores)
std_dev = np.std(zne_scores, ddof=1)
cv = (std_dev / mean_score) * 100

# Teste t 1-amostra sobre as diferenças (equivalente a comparar zne_scores vs baseline)
diff = zne_scores - greedy_baseline
t_stat, p_two_sided = stats.ttest_1samp(diff, 0.0)

# p unilateral para H1: mean(diff) > 0
df = len(diff) - 1
p_one_sided = 1.0 - stats.t.cdf(t_stat, df=df) if t_stat > 0 else 1.0

d_effect = cohens_d_onesample(zne_scores, greedy_baseline)
ci_lower, ci_upper = bootstrap_ci_mean(zne_scores)

improvement_pct = ((mean_score / greedy_baseline) - 1) * 100

# ==============================================================================
# 3. RELATÓRIO FINAL
# ==============================================================================
print("-" * 60)
print("STATISTICAL ANALYSIS REPORT (QAOA + ZNE Quadratic vs GREEDY)")
print("-" * 60)
print(f"Sample Size (n):            {len(zne_scores)} independent runs")
print(f"Mean ZNE Score:             {mean_score:.2f}")
print(f"Standard Deviation:         {std_dev:.2f}")
print(f"Coeff. Variation:           {cv:.2f}%")
print(f"Greedy Baseline:            {greedy_baseline:.2f}")
print("-" * 60)
print("One-sample t-test on differences (ZNE - Greedy):")
print(f"t({df}) = {t_stat:.2f}")
print(f"p (two-sided) = {p_two_sided:.4f}")
print(f"p (one-sided; H1: ZNE>Greedy) = {p_one_sided:.4f}")
print(f"Cohen's d (one-sample):      {d_effect:.2f}")
print(f"95% CI mean (bootstrap, B={BOOTSTRAP_B}): [{ci_lower:.2f}, {ci_upper:.2f}]")
print("-" * 60)

if p_one_sided < 0.05:
    print(f"RESULT: Statistically Significant Superiority (+{improvement_pct:.1f}%)")
else:
    print("RESULT: No significant superiority detected.")
print("-" * 60)
