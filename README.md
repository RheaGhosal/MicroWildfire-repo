#  Wildfire-Repro

**Reproducibility package for:**  
 *“AI-Driven Micro-Wildfire Prediction and Evacuation Planning Using Multimodal Data Fusion”*  
*(Rhea Ghosal et al.,nisurg.com, 2025)*

This repository contains the complete experimental pipeline used to reproduce all quantitative tables, fusion results, and fairness metrics from the paper.  
All code runs on open synthetic data—no private datasets required.

---

##  Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/wildfire-repo.git
cd wildfire-repo
2. Create environment and install dependencies
Option A — pip (recommended):

bash
Copy code
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt
Option B — Conda:

bash
Copy code
conda env create -f environment.yml
conda activate wildfire
3. Initialize packages (one-time setup)
bash
Copy code
python - << 'PY'
from pathlib import Path
Path("src/__init__.py").touch()
Path("scripts/__init__.py").touch()
print("Initialized packages: src/, scripts/")
PY
4. Run the complete pipeline
These commands reproduce all outputs from the paper.

macOS/Linux

bash
Copy code
export PYTHONPATH=$(pwd)
python -m scripts.split_data --config configs/config.yaml
python -m scripts.run_fusion --config configs/config.yaml
python -m scripts.run_bootstrap_ci --config configs/config.yaml
python -m scripts.reproduce_tables --config configs/config.yaml
Windows PowerShell

powershell
Copy code
$env:PYTHONPATH = (Get-Location).Path
python -m scripts.split_data --config configs/config.yaml
python -m scripts.run_fusion --config configs/config.yaml
python -m scripts.run_bootstrap_ci --config configs/config.yaml
python -m scripts.reproduce_tables --config configs/config.yaml
All outputs will appear in the out/ directory.

 Expected Outputs
bash
Copy code
out/
 ├─ splits.json          # deterministic 70/15/15 split (seed = 42)
 ├─ fusion_results.json   # performance metrics for fusion & baselines
 ├─ bootstrap_ci.json     # 95% confidence intervals for all metrics
 └─ fusion_table.csv      # main paper results table
Check them quickly:

bash
Copy code
python - << 'PY'
import pandas as pd, json
print(pd.read_csv("out/fusion_table.csv").head(), "\n")
print(json.load(open("out/bootstrap_ci.json")).keys())
PY
 Repository Structure
kotlin
Copy code
wildfire-repo/
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
    └─ (generated results)
📓 Run in Google Colab
python
Copy code
!git clone https://github.com/<your-username>/wildfire-repo.git
%cd wildfire-repo
!pip install -r requirements.txt

from pathlib import Path
Path("src/__init__.py").touch()
Path("scripts/__init__.py").touch()

import os
os.environ["PYTHONPATH"] = os.getcwd()

!python -m scripts.split_data --config configs/config.yaml
!python -m scripts.run_fusion --config configs/config.yaml
!python -m scripts.run_bootstrap_ci --config configs/config.yaml
!python -m scripts.reproduce_tables --config configs/config.yaml
Optionally open the interactive notebook:

bash
Copy code
notebooks/wild-fire.ipynb
 Configuration
You can modify:

configs/config.yaml → parameters, fusion type, metric settings.

src/metrics.py → add or edit metrics (e.g., EO/SPD/ΔFPR definitions).

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
 Run export PYTHONPATH=$(pwd) (Linux/macOS) or $env:PYTHONPATH=(Get-Location).Path (Windows).
Ensure both src/__init__.py and scripts/__init__.py exist.

 TypeError: keys must be str, int, float, bool or None, not int64
 Already patched. If you modify fairness metrics, cast dict keys:

python
Copy code
{k.item() if hasattr(k, "item") else k: v for k, v in data.items()}
 CalledProcessError when running scripts
 Run as modules, not plain scripts:

bash
Copy code
python -m scripts.run_fusion --config configs/config.yaml
 Citation
If you use this repository, please cite:

bibtex
Copy code
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
For questions or collaborations: [add email or GitHub contact]
