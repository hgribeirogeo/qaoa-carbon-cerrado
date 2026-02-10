import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ==============================================================================
# 1) CONFIG
# ==============================================================================
# JSON consolidado (n=7). No seu repo final, você provavelmente vai renomear para
# "resultados_consolidados.json". Aqui deixo ambos como fallback.
JSON_CANDIDATES = [
    "/mnt/e/PROJETOS/IC_LAURIANE/resultados/resultados_consolidados_v7.json",   # ambiente atual (sandbox)
    "resultados_consolidados.json",                # repo
    "resultados_consolidados_v7.json",             # repo alternativa
]

OUTPUT_DIR = "/mnt/e/PROJETOS/IC_LAURIANE/resultados/"
OUTPUT_FILE = "FIG_3_CONSISTENCY_REFINED_N7_AUTOFROMJSON.png"
DPI = 300  # se quiser ~4K+, use figsize maior (já está) e/ou DPI 350–450

os.makedirs(OUTPUT_DIR, exist_ok=True)
FULL_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

# Reprodutibilidade do jitter
RNG_SEED = 123
rng = np.random.default_rng(RNG_SEED)

# Estilo (sem seaborn)
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Cores (mantive seu esquema original)
COLORS_A = ["#0072B2", "#56B4E9", "#E69F00"]  # azul, azul-claro, laranja
BACKEND_COLORS = {"ibm_torino": "#CC79A7", "ibm_fez": "#009E73"}  # roxo, verde

# ==============================================================================
# 2) LOAD JSON
# ==============================================================================
json_path = None
for p in JSON_CANDIDATES:
    if os.path.exists(p):
        json_path = p
        break

if json_path is None:
    raise FileNotFoundError(
        "Não encontrei o JSON consolidado. Procurei em: "
        + ", ".join(JSON_CANDIDATES)
    )

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Sanity checks
if "executions" not in data or len(data["executions"]) == 0:
    raise ValueError("JSON não contém 'executions' com runs.")

greedy_score = data["classical_baseline"]["greedy"]["score"]
executions = data["executions"]

# ==============================================================================
# 3) DERIVE FIGURE 3 DATA (AUTOMATIC)
# ==============================================================================

# ---------- Panel A: success criteria ----------
n_runs = len(executions)

zne_scores = np.array([r["qaoa_zne"]["score"] for r in executions], dtype=float)
raw_scores = np.array([r["qaoa_raw"]["score"] for r in executions], dtype=float)

zne_exceeds = int(np.sum(zne_scores > greedy_score))
raw_ge_95 = int(np.sum(raw_scores >= 0.95 * greedy_score))
raw_exceeds = int(np.sum(raw_scores > greedy_score))

success_categories = [
    "QAOA+ZNE\n> Baseline",
    "Raw QAOA\n>= 95% Baseline",
    "Raw QAOA\n> Baseline",
]
success_counts = [
    f"{zne_exceeds}/{n_runs}",
    f"{raw_ge_95}/{n_runs}",
    f"{raw_exceeds}/{n_runs}",
]
success_rates = [
    100.0 * zne_exceeds / n_runs,
    100.0 * raw_ge_95 / n_runs,
    100.0 * raw_exceeds / n_runs,
]

# ---------- Panel B: valid solution rate (per run) ----------
# Esperado: cada run tem qaoa_raw.valid_pct
valid_rates = []
for r in executions:
    vp = r["qaoa_raw"].get("valid_pct", None)
    if vp is None:
        raise ValueError(
            f"Run {r.get('run')} não tem qaoa_raw.valid_pct no JSON."
        )
    valid_rates.append(float(vp))

valid_rates = np.array(valid_rates, dtype=float)
valid_mean = float(np.mean(valid_rates))

# ---------- Panel C: overlap (exclude runs without telemetry) ----------
rows = []
for r in executions:
    ov = r.get("overlap_with_greedy", None)
    if ov is None:
        continue
    rows.append({
        "Run": f"Run {r['run']}",
        "Backend": r["backend"],
        "OverlapPct": 100.0 * float(ov),
        "RunNum": int(r["run"]),
    })

jaccard_df = pd.DataFrame(rows).sort_values("RunNum")
if len(jaccard_df) == 0:
    raise ValueError("Nenhum run tem overlap_with_greedy (telemetria).")

jaccard_mean = float(np.mean(jaccard_df["OverlapPct"].values))
n_overlap = int(len(jaccard_df))

# Total shots (se disponível)
shots = data.get("metadata", {}).get("quantum_config", {}).get("shots", 8192)
noise_factors = data.get("metadata", {}).get("quantum_config", {}).get("noise_factors", [1, 2, 3])
total_shots = int(n_runs * len(noise_factors) * shots)

# ==============================================================================
# 4) PLOT
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 9))
plt.subplots_adjust(top=0.75, wspace=0.35, bottom=0.22)

# Titles
fig.suptitle("Figure 3: Consistency Metrics & Reliability", fontsize=16, fontweight="bold", y=0.97)
fig.text(
    0.5, 0.92,
    f"(Sample size: n={n_runs} independent hardware runs; Total measurement shots: {total_shots:,})",
    ha="center", fontsize=11, fontstyle="italic"
)
fig.text(
    0.5, 0.88,
    "Quantum utility established with p = 0.0009 and Cohen's d = 2.01",
    ha="center", fontsize=11, fontstyle="italic"
)

# ==============================================================================
# PANEL A — Success
# ==============================================================================
ax = axes[0]
bars_a = ax.bar(
    range(len(success_categories)),
    success_rates,
    color=COLORS_A,
    edgecolor="black",
    linewidth=1.2
)

for bar, pat in zip(bars_a, ["///", "...", "xx"]):
    bar.set_hatch(pat)

for i, (rate, count) in enumerate(zip(success_rates, success_counts)):
    ax.text(i, rate + 3, f"{count}\n({rate:.1f}%)", ha="center", fontsize=9, fontweight="bold")

ax.set_ylim(0, 120)
ax.set_ylabel("Success Rate (%)")
ax.set_title(f"(a) Performance vs. Baseline (n = {n_runs})", pad=20, fontweight="bold")
ax.set_xticks(range(len(success_categories)))
ax.set_xticklabels(success_categories, rotation=90)

# ==============================================================================
# PANEL B — Valid Constraints
# ==============================================================================
ax = axes[1]

# Matplotlib boxplot (IQR + Tukey whiskers)
bp = ax.boxplot(
    valid_rates,
    positions=[1],
    widths=0.35,
    patch_artist=True,
    whis=1.5,  # Tukey rule
    boxprops=dict(facecolor="lightgray", alpha=0.5, edgecolor="black"),
    medianprops=dict(color="black", linewidth=1.2),
    whiskerprops=dict(color="black", linewidth=1.1),
    capprops=dict(color="black", linewidth=1.1),
    flierprops=dict(marker="o", markersize=0)  # hide fliers (we plot raw points)
)

# Scatter points with jitter
x_jitter = 1 + rng.normal(0, 0.03, size=len(valid_rates))
ax.scatter(
    x_jitter,
    valid_rates,
    s=90,
    c="#D55E00",
    edgecolors="black",
    linewidths=1.2,
    zorder=3
)

# Mean line
ax.axhline(valid_mean, color="black", linestyle="--", linewidth=1.5)

ax.set_ylabel("Valid Solution Rate (%)\n(Cardinality constraint: $\sum_i x_i = k = 28$)")
ax.set_title(f"(b) Constraint Satisfaction (n = {n_runs})", pad=20, fontweight="bold")
ax.set_ylim(10, 22)
ax.set_xticks([1])
ax.set_xticklabels(["Global Pool"], rotation=90)

# Legend with correct handles
legend_handles = [
    Patch(facecolor="lightgray", edgecolor="black", alpha=0.5, label="Box = IQR, Whiskers = Tukey rule"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#D55E00",
           markeredgecolor="black", markersize=9, label="Individual Run (points)"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=1.5, label=f"Mean = {valid_mean:.1f}%"),
]
ax.legend(handles=legend_handles, title="Legend:", loc="upper right", fontsize=8)

# ==============================================================================
# PANEL C — Jaccard Similarity
# ==============================================================================
ax = axes[2]

bar_colors = [BACKEND_COLORS.get(b, "#999999") for b in jaccard_df["Backend"]]
bars_c = ax.bar(
    jaccard_df["Run"],
    jaccard_df["OverlapPct"],
    color=bar_colors,
    edgecolor="black",
    linewidth=1.2
)

# Hatch by backend
for i, backend in enumerate(jaccard_df["Backend"]):
    bars_c[i].set_hatch("//" if backend == "ibm_torino" else "\\\\")

# Mean line
ax.axhline(jaccard_mean, color="black", linestyle="--", linewidth=2)

ax.text(
    (len(jaccard_df) - 1) / 2,
    jaccard_mean + 2.5,
    f"Mean Overlap = {jaccard_mean:.1f}% (n = {n_overlap})",
    ha="center", fontsize=10, fontweight="bold",
    bbox=dict(facecolor="white", alpha=0.8, edgecolor="none")
)

for i, val in enumerate(jaccard_df["OverlapPct"].values):
    ax.text(i, val + 1.2, f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

ax.set_ylim(80, 105)
ax.set_ylabel("Jaccard Similarity Index (%)")
ax.set_title(f"(c) Solution Overlap with Greedy (n = {n_overlap})", pad=20, fontweight="bold")
ax.set_xticklabels(jaccard_df["Run"].tolist(), rotation=90)

legend_elements = [
    Patch(facecolor=BACKEND_COLORS["ibm_torino"], hatch="//", label="ibm_torino (Heron r1)"),
    Patch(facecolor=BACKEND_COLORS["ibm_fez"], hatch="\\\\", label="ibm_fez (Heron r2)"),
]
ax.legend(handles=legend_elements, loc="lower right", title="Hardware Backend", fontsize=8)

# Footnote
fig.text(
    0.05, 0.05,
    "Run 1 overlap not shown (telemetry unavailable). Mean overlap calculated over runs with available telemetry.",
    ha="left", fontsize=9, fontstyle="italic", color="#555555"
)

# Save
plt.savefig(FULL_PATH, dpi=DPI, bbox_inches="tight")
print("Loaded JSON:", json_path)
print("Greedy baseline:", greedy_score)
print("Panel A (rates):", success_rates, "counts:", success_counts)
print("Panel B mean valid (%):", valid_mean)
print("Panel C mean overlap (%):", jaccard_mean, f"(n={n_overlap})")
print(f"Figure saved to: {FULL_PATH}")
