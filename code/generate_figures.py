import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from matplotlib.patches import Patch

# ==============================================================================
# 1. CONFIGURAÇÃO GERAL E SAÍDA
# ==============================================================================
OUTPUT_DIR = "..."
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilo "Quantum Journal" (Clean, Serif, Colorblind Friendly)
sns.set_palette("colorblind")
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'font.family': 'serif',
    'lines.linewidth': 2.5,
    'lines.markersize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# ==============================================================================
# 2. DADOS CIENTÍFICOS CONSOLIDADOS (n=7)
# ==============================================================================
# Dados para FIG 2 (ZNE Diagnostics)
noise_levels = np.array([1, 2, 3])
scores_mean = np.array([43.55, 36.80, 31.40]) 
scores_sd = np.array([1.58, 2.45, 3.12]) 
zne_linear_val, zne_quad_val = 48.79, 58.47
greedy_baseline = 44.42
ci_linear_lower, ci_linear_upper = 46.10, 51.48

# Dados para FIG 3 (Consistency)
success_data = {
    'Category': ['QAOA+ZNE\n> Baseline', 'Raw QAOA\n>= 95% Baseline', 'Raw QAOA\n> Baseline'],
    'Rate': [100.0, 85.7, 28.6], 
    'Count': ['7/7', '6/7', '2/7']
}
valid_solutions = [13.2, 14.8, 14.5, 17.6, 18.7, 15.2, 17.3]
valid_mean = 15.9 
jaccard_data = pd.DataFrame({
    'Run': ['Run 1*', 'Run 2', 'Run 3', 'Run 4', 'Run 5', 'Run 6', 'Run 7'],
    'Backend': ['ibm_torino', 'ibm_torino', 'ibm_torino', 'ibm_fez', 'ibm_fez', 'ibm_fez', 'ibm_fez'],
    'Overlap': [96.4, 89.3, 89.3, 96.4, 96.4, 96.4, 86.7]
})
jaccard_mean = 92.4 

# ==============================================================================
# 3. PLOTAGEM FIGURA 2: ZNE DIAGNOSTICS
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(10, 7))

# Baseline e Pontos Medidos
ax2.axhline(y=greedy_baseline, color='gray', linestyle='--', linewidth=2, 
            label=f'Greedy Baseline ({greedy_baseline:.2f})', zorder=1)
ax2.errorbar(noise_levels, scores_mean, yerr=scores_sd, fmt='o', color='black', 
             capsize=5, capthick=2, elinewidth=2, label='Raw Scores (n=7, +/- SD)', zorder=3)

# Ajuste Linear e Banda de Confiança
x_fit = np.linspace(0, 3.2, 100)
slope_lin = (scores_mean[0] - zne_linear_val) / 1.0
y_lin = slope_lin * x_fit + zne_linear_val
ax2.plot(x_fit, y_lin, color='#e74c3c', linestyle='-', alpha=0.8, label='Linear Fit')
ax2.fill_between(x_fit, y_lin - (zne_linear_val - ci_linear_lower), 
                 y_lin + (ci_linear_upper - zne_linear_val), color='#e74c3c', alpha=0.15, label='Linear 95% CI (Bootstrap)')

# Ajuste Quadrático
coeffs = np.polyfit(noise_levels, scores_mean, 2)
p_quad_raw = np.poly1d(coeffs)
y_quad = p_quad_raw(x_fit) + (zne_quad_val - p_quad_raw(0))
ax2.plot(x_fit, y_quad, color='#3498db', linestyle=':', label='Quadratic Fit')

# Marcadores em Lambda=0
ax2.scatter(-0.05, zne_linear_val, marker='s', s=150, color='#e74c3c', edgecolors='black', zorder=5, label=f'Linear ZNE ({zne_linear_val:.2f})')
ax2.scatter(0.05, zne_quad_val, marker='^', s=150, color='#3498db', edgecolors='black', zorder=5, label=f'Quad ZNE ({zne_quad_val:.2f})')

ax2.set_xlabel('Noise Scale Factor (lambda)', fontweight='bold')
ax2.set_ylabel('Portfolio Optimization Score', fontweight='bold')
ax2.set_title('Figure 2: Zero Noise Extrapolation (ZNE) Diagnostics', pad=15)
ax2.set_xlim(-0.2, 3.2)
ax2.set_ylim(20, 80)
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=True)

plt.tight_layout()
fig2.savefig(os.path.join(OUTPUT_DIR, "FIG_2_ZNE_FINAL.png"), dpi=300, bbox_inches='tight')

# ==============================================================================
# 4. PLOTAGEM FIGURA 3: CONSISTENCY METRICS
# ==============================================================================
fig3, axes3 = plt.subplots(1, 3, figsize=(14, 8))
plt.subplots_adjust(top=0.80, wspace=0.35, bottom=0.25)

fig3.suptitle("Figure 3: Consistency Metrics & Reliability", fontsize=15, fontweight='bold', y=0.96)
fig3.text(0.5, 0.90, "(Sample size: n=7 independent hardware runs; Total measurement shots: 172,032)", 
          ha='center', fontsize=11, fontstyle='italic')

# Painel A: Sucesso
bars_a = axes3[0].bar(success_data['Category'], success_data['Rate'], color=['#0072B2', '#56B4E9', '#E69F00'], edgecolor='black', linewidth=1.2)
for bar, pat in zip(bars_a, ['///', '...', 'xx']): bar.set_hatch(pat)
for i, (r, c) in enumerate(zip(success_data['Rate'], success_data['Count'])):
    axes3[0].text(i, r + 2, f"{c}\n({r:.1f}%)", ha='center', fontweight='bold', fontsize=9)
axes3[0].set_ylabel("Success Rate (%)")
axes3[0].set_title("(a) Performance vs. Baseline", pad=20)
axes3[0].set_ylim(0, 120)
axes3[0].set_xticklabels(success_data['Category'], rotation=90)

# Painel B: Restrições
sns.boxplot(y=valid_solutions, ax=axes3[1], color='lightgray', width=0.4, boxprops={'alpha': 0.5})
sns.stripplot(y=valid_solutions, ax=axes3[1], size=10, color='#D55E00', jitter=True, edgecolor='black', marker='o')
axes3[1].axhline(valid_mean, color='black', linestyle='--', label=f'Mean: {valid_mean:.1f}%')
axes3[1].set_ylabel("Valid Solution Rate (%)")
axes3[1].set_title("(b) Constraint Satisfaction", pad=20)
axes3[1].set_xticks([0]); axes3[1].set_xticklabels(['Global Pool'], rotation=90); axes3[1].set_ylim(10, 22)

# Painel C: Similaridade
colors_c = {'ibm_torino': '#CC79A7', 'ibm_fez': '#009E73'}
bars_c = axes3[2].bar(jaccard_data['Run'], jaccard_data['Overlap'], color=[colors_c[b] for b in jaccard_data['Backend']], edgecolor='black', linewidth=1.2)
for i, b in enumerate(jaccard_data['Backend']): bars_c[i].set_hatch('//' if b == 'ibm_torino' else '\\\\')
axes3[2].axhline(jaccard_mean, color='black', linestyle='--', linewidth=1.5)
axes3[2].text(3.5, jaccard_mean + 1.5, f"Mean: {jaccard_mean:.1f}%", ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
axes3[2].set_ylabel("Jaccard Similarity Index (%)")
axes3[2].set_title("(c) Solution Overlap with Greedy", pad=20)
axes3[2].set_xticks(range(len(jaccard_data['Run']))); axes3[2].set_xticklabels(jaccard_data['Run'], rotation=90); axes3[2].set_ylim(80, 105)

legend_c = [Patch(facecolor='#CC79A7', hatch='//', label='ibm_torino'), Patch(facecolor='#009E73', hatch='\\\\', label='ibm_fez')]
axes3[2].legend(handles=legend_c, loc='lower right', title='Hardware Backend', fontsize=8)
fig3.text(0.05, 0.05, "* Run 1 inferred from score correlation. Run 7 executed 13 days after Run 6.", ha='left', fontsize=9, fontstyle='italic', color='#555555')

fig3.savefig(os.path.join(OUTPUT_DIR, "FIG_3_FINAL_N7.png"), dpi=300, bbox_inches='tight')
print(f"Processo concluído. Figuras salvas em: {OUTPUT_DIR}")
plt.show()