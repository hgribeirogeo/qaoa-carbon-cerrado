QAOA with Zero Noise Extrapolation for Carbon Credit Portfolio Optimization
Quantum Approximate Optimization Algorithm (QAOA) with Zero Noise Extrapolation (ZNE) applied to carbon credit portfolio optimization in the Brazilian Cerrado biome.

🔗 Paper: Submitted to Quantum Journal | arXiv: PENDING
👤 Author: Hugo José Ribeiro (Federal University of Goiás, Brazil)
📧 Contact: hugoppgema@ufg.br

📊 Key Results

100% success rate across 7 independent runs on IBM Quantum hardware (7/7)

31.6% average improvement over the classical Greedy baseline (t(6)=5.33, p=0.0009; one-sided Wilcoxon W=28, p=0.0078; Cohen’s d=2.01)

Problem scale: n = 88 municipalities, k = 28 selection, C(88,28) ≈ 1.45×10²² portfolios

Hardware: ibm_torino (Heron r1, 133 qubits), ibm_fez (Heron r2, 156 qubits)

Execution period: January 17 – February 2, 2026

Total measurement shots: 172,032 (7 runs × 3 noise factors × 8,192 shots)

🚀 Quick Start
Installation
git clone https://github.com/hgribeirogeo/qaoa-carbon-cerrado.git
cd qaoa-carbon-cerrado
pip install -r requirements.txt

Reproduce Results
# Coming soon
python code/qaoa_implementation.py
python code/generate_figures.py
python code/zne_analysis.py

📁 Repository Structure
├── data/              # Municipal scores, adjacency matrices, synergy data
├── code/              # QAOA implementation, ZNE protocol, baselines, figures, analysis
├── results/           # Experimental data from IBM Quantum (7 runs)
│   ├── resultados_consolidados_v7.json
│   └── figures/       # Paper figures
└── paper/             # Manuscript and supplementary materials

📈 Results Summary
Method	Score	vs Greedy	Success Rate
Greedy	44.42	100.0%	baseline
Simulated Annealing	42.23 ± 0.51	95.1%	0/7
QAOA (raw)	43.55 ± 1.54	98.0%	2/7
QAOA + ZNE	58.47 ± 6.98	131.6%	7/7

Solution consistency: 92.4% mean overlap with Greedy across Runs 2–7 (n = 6)
Constraint satisfaction: 15.9% mean feasible-shot rate (cardinality satisfied)
Temporal stability: Run 7 confirmed consistent performance after a multi-day calibration interval.

🔬 Methodology
Problem Formulation

Multi-objective QUBO: carbon sequestration + biodiversity + social impact

n = 88 municipalities in Goiás state (Cerrado biome)

k = 28 selection (fixed-cardinality portfolio)

Weights: w_C = 0.33, w_B = 0.33, w_S = 0.34

QAOA Implementation

Depth: p = 1 (~250 native gates)

Warm-start: initialization from Greedy solution

Mixer: XY-type (number-conserving in the ideal unitary limit) with a retained quadratic cardinality penalty to mitigate noise-induced violations

Feasible-shot rate (empirical): 15.9%

Zero Noise Extrapolation

Gate folding on 2-qubit operations

Noise amplification: λ ∈ {1, 2, 3}

Shots: 8,192 per level (24,576 per run)

Extrapolation methods: linear, quadratic, Richardson

Uncertainty quantification: bootstrap (B = 100, 95% CI)

💾 Data Description
Input Data (data/)

Municipal scores (88 municipalities):

Carbon sequestration potential (MapBiomas + GEDI/LiDAR)

Biodiversity indicators (endemic species, conservation units)

Social impact metrics (rural population, vulnerability)

Spatial relationships:

Adjacency matrix (88×88)

Biodiversity synergy matrix

Social synergy matrix

Output Data (results/)

Complete experimental results: results/resultados_consolidados_v7.json

7 independent runs (3× ibm_torino, 4× ibm_fez)

Raw measurements (λ = 1) + amplified circuits (λ = 2, 3)

ZNE extrapolations (linear, quadratic, Richardson)

Bootstrap confidence intervals

Reproducibility metadata:

IBM Quantum job IDs

Execution timestamps

Backend details (as logged in the consolidated results)

📚 Citation

If you use this code or data, please cite:

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


DOI: 10.5281/zenodo.18418054 (Zenodo archive)

🔗 Related Projects

Atlas Biomassa Goiás
 - Biomass estimation model (R²=0.77)

Interactive Dashboard
 - Carbon data visualization

📄 License

This project is licensed under the MIT License - see LICENSE
 file for details.

🙏 Acknowledgments

We acknowledge the use of IBM Quantum services for this work. The views expressed are those of the author and do not reflect the official policy or position of IBM or IBM Quantum.

📞 Contact

Hugo José Ribeiro
Federal University of Goiás (UFG), Brazil
School of Civil and Environmental Engineering
Email: hugoppgema@ufg.br

GitHub: @hgribeirogeo

---

**Last updated:** January 2026
