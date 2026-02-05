# QAOA with Zero Noise Extrapolation for Carbon Credit Portfolio Optimization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18418054.svg)](https://doi.org/10.5281/zenodo.18418054)
[![arXiv](https://img.shields.io/badge/arXiv-PENDING-b31b1b.svg)](https://arxiv.org/abs/PENDING)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Quantum Journal](https://img.shields.io/badge/Quantum-Submitted-blue.svg)](https://quantum-journal.org)

**Empirical Evaluation of QAOA with Zero Noise Extrapolation on NISQ Hardware for Carbon Credit Portfolio Optimization in the Brazilian Cerrado.**

🔗 **Paper:** Submitted to *Quantum Journal* | arXiv: **PENDING**  
👤 **Author:** Hugo José Ribeiro (Federal University of Goiás, Brazil)  
📧 **Contact:** hugoppgema@ufg.br  

---

## 📊 Key Results

- ✅ **100% success rate** across **7 independent hardware runs** (7/7)
- 📈 **31.6% average improvement** over the Greedy baseline  
  *(t(6)=5.33, p=0.0009; Wilcoxon W=28, p=0.0078; Cohen's d=2.01)*
- 🧮 **Problem scale:** n = 88 municipalities, k = 28 selection  
  *(C(88,28) ≈ 1.45 × 10²² portfolios)*
- 🖥️ **Hardware:**  
  - `ibm_torino` (Heron r1, 133 qubits)  
  - `ibm_fez` (Heron r2, 156 qubits)
- 📅 **Execution period:** January 17 – February 2, 2026  
- 🎯 **Total measurement shots:** 172,032

---

## 🚀 Quick Start

### 🔧 Installation
```bash
git clone https://github.com/hgribeirogeo/qaoa-carbon-cerrado.git
cd qaoa-carbon-cerrado
pip install -r requirements.txt
```

### ▶️ Reproduce Results
```bash
# Scripts will be finalized after publication
python code/qaoa_implementation.py
python code/check_article_metrics.py
python code/figure_2.py
python code/figure_3.py
```

---

## 📁 Repository Structure

```
├── data/              # Municipal scores, adjacency matrices, synergy data
├── code/              # QAOA implementation, check metrics, figures
├── results/           # Experimental data from IBM Quantum (7 runs)
│   ├── resultados_consolidados_v7.json
│   └── figures/       # Figures used in the paper
└── paper/             # Manuscript and supplementary materials
```

---

## 📈 Results Summary

| Method | Score | vs. Greedy | Success Rate |
|--------|-------|------------|--------------|
| Greedy | 44.42 | 100.0% | baseline |
| Simulated Annealing | 42.23 ± 0.51 | 95.1% | 0 / 7 |
| QAOA (raw) | 43.55 ± 1.54 | 98.0% | 2 / 7 |
| **QAOA + ZNE** | **58.47 ± 6.98** | **131.6%** | **7 / 7** |

📌 **Consistency:**
- Mean overlap with Greedy: 92.4% (Runs 2–7, n = 6)
- Mean feasible-shot rate: 15.9% (cardinality satisfied)

---

## 🔬 Methodology

### 🧩 Problem Formulation

- **Multi-objective QUBO:**  
  Carbon sequestration + Biodiversity + Social impact
- **n = 88 municipalities** (Goiás state, Cerrado biome)
- **k = 28 fixed-cardinality** selection
- **Weights:** w_C = 0.33, w_B = 0.33, w_S = 0.34

### ⚙️ QAOA Implementation

- **Depth:** p = 1 (~250 native gates)
- **Warm-start:** initialized from Greedy solution
- **Mixer:** XY-type (number-conserving ideally)  
  with retained quadratic penalty for noise robustness
- **Empirical feasible-shot rate:** 15.9%

### 🔇 Zero Noise Extrapolation (ZNE)

- **Gate folding:** two-qubit gates
- **Noise scaling:** λ ∈ {1, 2, 3}
- **Shots:** 8,192 per λ (24,576 per run)
- **Extrapolation:** linear, quadratic, Richardson
- **Uncertainty:** bootstrap (B = 100, 95% CI)

---

## 💾 Data Description

### 📥 Input Data (`data/`)

**Municipal attributes (88 units):**
- Carbon potential (MapBiomas + GEDI/LiDAR)
- Biodiversity indicators
- Social vulnerability metrics

**Spatial structure:**
- Adjacency matrix (88 × 88)
- Biodiversity and social synergy matrices

### 📤 Output Data (`results/`)

**Consolidated results:**  
`results/resultados_consolidados_v7.json`

- 7 independent hardware runs
- Raw and noise-amplified circuits
- ZNE extrapolations
- Bootstrap confidence intervals

**Metadata:**
- IBM Quantum job IDs, timestamps, backend info

---

## 📚 Citation

If you use this repository, please cite:

```bibtex
@article{Ribeiro2026QAOA,
  title={Empirical Evaluation of QAOA with Zero Noise Extrapolation on NISQ Hardware
                   for Carbon Credit Portfolio Optimization in the Brazilian Cerrado},
  author={Ribeiro, Hugo Jos{\'e}},
  journal={Quantum},
  year={2026},
  note={Submitted},
  archivePrefix={arXiv},
  eprint={PENDING}
}
```

🔗 **Zenodo DOI:** https://doi.org/10.5281/zenodo.18418054

---

## 🔗 Related Projects

🌱 **Atlas Biomassa Goiás**  
https://github.com/hgribeirogeo/atlas-biomassa-goias

📊 **Interactive Dashboard**  
https://atlas-biomassa-goias.streamlit.app/

---

## 📄 License

This project is licensed under the MIT License – see [LICENSE](LICENSE).

---

## 🙏 Acknowledgments

We acknowledge the use of IBM Quantum services for this work.  
The views expressed are those of the author and do not necessarily reflect the official policies of IBM or IBM Quantum.

---

## 📞 Contact

**Hugo José Ribeiro**  
Federal University of Goiás (UFG), Brazil  
School of Civil and Environmental Engineering  
📧 hugoppgema@ufg.br  

🐙 GitHub: [@hgribeirogeo](https://github.com/hgribeirogeo)

---

**Last updated:** February 2026
