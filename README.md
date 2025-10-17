# Real-Time Micro-Wildfire Risk Forecasting — Reproducibility Pack

This repository reproduces key tables/figures from:
**"AI-Driven Micro-Wildfire Prediction and Evacuation Planning Using Multimodal Data Fusion".**

It addresses reviewer concerns on **(1) multimodal fusion evidence**, **(2) fairness/statistical rigor (EO/SPD/ΔFPR with CIs + calibration)**, **(3) calibration metrics (ECE/Brier)**, and **(4) documented evaluation protocol**:

- **70/15/15 stratified split** with `random_state: 42`.
- **Youden’s J**-selected threshold on validation, applied unchanged to test.
- **Bootstrapped 95% CIs** (default B=200; paired by region ID where available).
- **Fusion** via **weighted averaging** and **stacking (logistic regression)** with calibrated probabilities.
- **Fairness metrics** across Köppen–Geiger climate zones (proxy groups), plus ECE/Brier overall and by subgroup.

> ⚠️ **Data**: Use your local copy of public datasets (FIRMS/MODIS/VIIRS/Sentinel, weather/terrain). Place files under `data/` and edit `configs/config.yaml`. To run without real data, see the **synthetic demo** below that exercises the full pipeline.

## Quickstart
```bash
# (A) Create environment
conda env create -f environment.yml  # or: python -m venv .venv && pip install -r requirements.txt
conda activate wildfire-repro

# (B) Configure data paths
cp configs/config.example.yaml configs/config.yaml
# edit configs/config.yaml to point to your local data

# (C) Split, train, calibrate, evaluate
python scripts/split_data.py --config configs/config.yaml
python scripts/run_fusion.py --config configs/config.yaml
python scripts/run_bootstrap_ci.py --config configs/config.yaml

# (D) Reproduce tables (stdout/CSV in out/)
python scripts/reproduce_tables.py --config configs/config.yaml
```

## Synthetic demo (no data required)
```bash
python scripts/run_synthetic_demo.py
```
This generates synthetic tabular/image/sequence features and runs the evaluation protocol to produce AUROC/Brier/ECE and fairness metrics with CIs, validating end-to-end logic.

## Layout
```
wildfire-repro/
├─ README.md
├─ environment.yml / requirements.txt
├─ configs/
│  └─ config.example.yaml
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
├─ notebooks/
│  └─ 01_reproduce_tables.ipynb
├─ CITATION.cff
├─ LICENSE
└─ Makefile
```

## Outputs
- `out/metrics_*.csv` — AUROC/Brier/ECE per model with 95% CIs.
- `out/fairness_ci.csv` — EO/SPD/ΔFPR pre/post calibration with 95% CIs.
- `out/fusion_delta_auc.csv` — paired ΔAUROC vs best single model.
- `out/acc_f1_ci.csv` — Accuracy/F1 with 95% CIs for VAL-chosen threshold.

## Reviewer-facing checklist (copy to your paper)
- Exact split policy and seeds documented (`random_state: 42`).
- Thresholding policy documented (Youden’s J on VAL, fixed on TEST).
- Bootstrap protocol documented (B=200; paired by region if available).
- Fusion method, calibration method, and fairness groups defined.
- Scripts provided to regenerate tables; unit tests for metrics correctness.

## How to cite
See `CITATION.cff`.
