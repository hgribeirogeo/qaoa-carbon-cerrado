# QAOA with Zero Noise Extrapolation for Carbon Credit Portfolio Optimization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18418054.svg)](https://doi.org/10.5281/zenodo.18418054)
[![arXiv](https://img.shields.io/badge/arXiv-PENDING-b31b1b.svg)](https://arxiv.org/abs/PENDING)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Quantum Journal](https://img.shields.io/badge/Quantum-Submitted-blue.svg)](https://quantum-journal.org)

**Quantum Approximate Optimization Algorithm (QAOA) with Zero Noise Extrapolation (ZNE) applied to carbon credit portfolio optimization in the Brazilian Cerrado biome.**

🔗 **Paper:** Submitted to Quantum Journal | arXiv: PENDING  
👤 **Author:** Hugo José Ribeiro (Universidade Federal de Goiás)  
📧 **Contact:** hugoppgema@ufg.br

---

## 📊 Key Results

- **100% success rate** across 7 independent runs on IBM Quantum hardware
- **31.6% average improvement** over classical greedy baseline (p < 0.0008)
- **Problem scale:** n=88 municipalities, k=28 selection, ~10²² combinations
- **Hardware:** ibm_torino (Heron r1, 133q), ibm_fez (Heron r2, 156q)
- **Execution period:** January 17 – February 2, 2026

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/hgribeirogeo/qaoa-carbon-cerrado.git
cd qaoa-carbon-cerrado
pip install -r requirements.txt
```

### Reproduce Results
```python
# Coming soon
python code/qaoa_implementation.py
python code/generate_figures.py
python code/zne_analysis.py
```

---

## 📁 Repository Structure
```
├── data/              # Municipal scores, adjacency matrices, synergy data
├── code/              # QAOA implementation, ZNE protocol, baselines
├── results/           # Experimental data from IBM Quantum (6 runs)
│   └── figures/       # Paper figures
└── paper/             # Manuscript and supplementary materials
```

---

## 📈 Results Summary

| Method              | Score          | vs Greedy | Success Rate |
|---------------------|----------------|-----------|--------------|
| Greedy              | 44.42          | 100.0%    | baseline     |
| Simulated Annealing | 42.23 ± 0.51   | 95.1%     | 0/7          |
| QAOA (raw)          | 43.55 ± 1.58   | 98.4%     | 2/7          |
| **QAOA + ZNE**      | **58.47 ± 6.54** | **131.6%** | **7/7**    |

**Solution consistency:** 92.4% average overlap between runs
**Temporal Stability:** Run 7 confirmed consistent performance after a 13-day hardware calibration interval.

---

## 🔬 Methodology

### Problem Formulation
- Multi-objective QUBO: carbon sequestration + biodiversity + social impact
- **n = 88** municipalities in Goiás state (Cerrado biome)
- **k = 28** selection constraint
- Weights: w_C=0.33, w_B=0.33, w_S=0.34

### QAOA Implementation
- **Depth:** p=1 (~250 native gates)
- **Warm-start:** initialization from greedy solution
- **Mixer:** Standard transverse field with quadratic penalty for cardinality.
- **Valid solution rate:** 15.9%

### Zero Noise Extrapolation
- **Gate folding** on 2-qubit gates (CNOT, RZZ)
- **Noise amplification:** λ ∈ {1, 2, 3}
- **Shots:** 8,192 per level (24,576 total per run)
- **Extrapolation methods:** linear, quadratic, Richardson
- **Uncertainty quantification:** bootstrap (B=100, 95% CI)

---

## 💾 Data Description

### Input Data (`data/`)

- **Municipal scores** (88 municipalities):
  - Carbon sequestration potential (MapBiomas + GEDI LiDAR)
  - Biodiversity indicators (endemic species, conservation units)
  - Social impact metrics (rural population, vulnerability)
  
- **Spatial relationships:**
  - Adjacency matrix (88×88)
  - Biodiversity synergy matrix
  - Social synergy matrix

### Output Data (`results/`)

- **Complete experimental results:**
  - 7 independent runs (3× ibm_torino, 4× ibm_fez)
  - Raw measurements (λ=1) + amplified circuits (λ=2,3)
  - ZNE extrapolations (linear, quadratic, Richardson)
  - Bootstrap confidence intervals
  
- **Reproducibility:**
  - IBM Quantum job IDs
  - Execution timestamps
  - Backend calibration data

---

## 📚 Citation

If you use this code or data, please cite:
```bibtex
@article{Ribeiro2026QAOA,
  title={QAOA with Zero Noise Extrapolation Outperforms Classical Heuristics 
         for Carbon Credit Portfolio Optimization in Brazilian Cerrado},
  author={Ribeiro, Hugo Jos{\'e}},
  journal={Quantum},
  year={2026},
  note={Submitted},
  archivePrefix={arXiv},
  eprint={PENDING}
}
```

**DOI:** 10.5281/zenodo.18418054 (Zenodo archive)

---

## 🔗 Related Projects

- [Atlas Biomassa Goiás](https://github.com/hgribeirogeo/atlas-biomassa-goias) - Biomass estimation model (R²=0.77)
- [Interactive Dashboard](https://atlas-biomassa-goias.streamlit.app/) - Carbon data visualization

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

We acknowledge the use of IBM Quantum services for this work. The views expressed are those of the author and do not reflect the official policy or position of IBM or IBM Quantum.

---

## 📞 Contact

**Hugo José Ribeiro**  
Universidade Federal de Goiás  
Departamento de Gestão e Geomática  
Email: hugoppgema@ufg.br  
GitHub: [@hgribeirogeo](https://github.com/hgribeirogeo)

---

**Last updated:** January 2026
