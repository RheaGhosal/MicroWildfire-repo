#  Wildfire-Repro

**Reproducibility package for:**  
 *“AI-Driven Micro-Wildfire Prediction and Evacuation Planning Using Multimodal Data Fusion”*  
*(Rhea Ghosal et al., IEEE Access, 2025)*

This repository contains the complete experimental pipeline used to reproduce all quantitative tables, fusion results, and fairness metrics from the paper.  
All code runs on open synthetic data—no private datasets required.
>  **Tested Environment:** macOS- Sequoia 15.7.1 and Ubuntu 22.04 (Python 3.11).  
> Verified on both platforms with identical results. For Windows, use WSL 2.



##  Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/RheaGhosal/MicroWildfire-repo.git
cd MicroWildfire-repo
2. Create environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
pip install -r requirements.txt
3. Initialize packages (one-time setup)
python - << 'PY'
from pathlib import Path
Path("src/__init__.py").touch()
Path("scripts/__init__.py").touch()
print("Initialized packages: src/, scripts/")
PY
4. Run the complete pipeline

#macOS/Linux
export PYTHONPATH=$(pwd)
python -m scripts.split_data --config configs/config.yaml
python -m scripts.run_fusion --config configs/config.yaml
# Choose which model to compute confidence intervals for:
# (Default = stacking)
python -m scripts.run_bootstrap_ci --config configs/config.yaml --model stacking

# Or for weighted fusion:
# python -m scripts.run_bootstrap_ci --config configs/config.yaml --model weighted

# Or for the best single-modality baseline:
# python -m scripts.run_bootstrap_ci --config configs/config.yaml --model best

python -m scripts.reproduce_tables --config configs/config.yaml

Tip: To keep separate CI files per model, rename after each run:

mv out/bootstrap_ci.json out/bootstrap_ci_stacking.json
mv out/bootstrap_ci.json out/bootstrap_ci_weighted.json

Expected Outputs

out/
 ├─ splits.json          # deterministic 70/15/15 split (seed = 42)
 ├─ fusion_results.json   # performance metrics for fusion & baselines
 ├─ bootstrap_ci.json     # 95% confidence intervals for all metrics
 └─ fusion_table.csv      # main paper results table

Quickly preview results inside the environment
python - << 'PY'
import pandas as pd, json
print(pd.read_csv("out/fusion_table.csv").head(), "\n")
print(json.load(open("out/bootstrap_ci.json")).keys())
PY
 Repository Structure

MicroWildfire-repo/
├─ README.md
├─ LICENSE
├─ CITATION.cff
├─ environment.yml
├─ configs/
│  └─ config.yaml
├─ src/
│  ├─ data.py
│  ├─ models.py
│  ├─ calibration.py
│  ├─ metrics.py
│  ├─ fusion.py
│  └─ evaluation.py
├─ scripts/
│  ├─ split_data.py
│  ├─ run_fusion.py
│  ├─ run_bootstrap_ci.py
│  ├─ reproduce_tables.py
│  └─ run_synthetic_demo.py
├─ tests/
│  └─ test_metrics.py
└─ out/
│  ├─ bootstrap_ci.json
│  ├─ fusion_results.json
│  ├─ fusion_table.cs
│  ├─ reproduce_tables.py
│  └─ splits.json

 What Each Script Does
Script	Description
scripts/split_data.py	Creates fixed stratified split and saves splits.json.
scripts/run_fusion.py	Trains baseline and fusion models; computes AUROC/Brier/ECE.
scripts/run_bootstrap_ci.py	Bootstraps confidence intervals (default B = 200).
scripts/reproduce_tables.py	Compiles and formats paper tables.
src/metrics.py	Implements fairness and calibration metrics.
src/fusion.py	Implements weighted and stacking fusion.

 Reproducibility Protocol
Split: 70/15/15 stratified (seed = 42)

Thresholding: single validation-selected threshold (Youden’s J)

Metrics: AUROC, Brier score, ECE, Accuracy, F1

Fairness: EO, SPD, ΔFPR with bootstrap 95% CIs

Fusion: weighted average & stacking vs best single modality

 Troubleshooting
 ModuleNotFoundError: No module named 'src'
 Run export PYTHONPATH=$(pwd) (Linux/macOS) 
Ensure both src/__init__.py and scripts/__init__.py exist.

 TypeError: keys must be str, int, float, bool or None, not int64.
 If you modify fairness metrics, cast dict keys:

{k.item() if hasattr(k, "item") else k: v for k, v in data.items()}
 CalledProcessError when running scripts
 Run as modules, not plain scripts:

python -m scripts.run_fusion --config configs/config.yaml

 Citation
If you use this repository, please cite:

bibtex

@article{ghosal2025wildfire,
  title   = {AI-Driven Micro-Wildfire Prediction and Evacuation Planning Using Multimodal Data Fusion},
  author  = {Rhea Ghosal and others},
  journal = {IEEE Access},
  year    = {2025}
}
 License
This project is licensed under the MIT License.
See the LICENSE file for details.

Acknowledgment
This repository accompanies the IEEE Access paper submission
and is maintained by Rhea Ghosal (Westlake High School, TX, USA).
For questions or collaborations: [[GitHub contact]](https://github.com/RheaGhosal/MicroWildfire-repo/issues)


Note: This repository was publicly released in October 2025 to accompany the revised IEEE Access submission. 
It reflects the exact scripts and metrics used to produce Tables 3–6 and fairness results in the final manuscript.
