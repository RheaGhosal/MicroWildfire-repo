import argparse, yaml, os, json
import numpy as np
from src.data import load_synthetic
from src.metrics import bootstrap_ci, auroc, brier, ece
from src.metrics import bootstrap_parity_cis  # EO/SPD/ΔFPR with CIs

def main(cfg):
    os.makedirs("out", exist_ok=True)

    # Synthetic placeholders; swap to your real probs later if needed
    y, p_tab, p_seq, p_img, groups = load_synthetic()

    B = cfg["bootstrap"]["B"]

    # Overall metric CIs (example on tabular stream)
    mean_auc,  lo_auc,  hi_auc  = bootstrap_ci(auroc,  y, p_tab, B=B)
    mean_bri,  lo_bri,  hi_bri  = bootstrap_ci(brier,  y, p_tab, B=B)
    mean_ece,  lo_ece,  hi_ece  = bootstrap_ci(ece,    y, p_tab, B=B)

    # Fairness CIs (EO/SPD/ΔFPR) at threshold=0.5 (adjust if you want VAL-chosen thr)
    fair = bootstrap_parity_cis(y, p_tab, groups, threshold=0.5, B=B)

    out = dict(
        auroc=[float(mean_auc), float(lo_auc), float(hi_auc)],
        brier=[float(mean_bri), float(lo_bri), float(hi_bri)],
        ece=[float(mean_ece), float(lo_ece), float(hi_ece)],
        eo=dict(mean=float(fair["eo"]["mean"]), lo=float(fair["eo"]["lo"]), hi=float(fair["eo"]["hi"])),
        spd=dict(mean=float(fair["spd"]["mean"]), lo=float(fair["spd"]["lo"]), hi=float(fair["spd"]["hi"])),
        delta_fpr=dict(mean=float(fair["delta_fpr"]["mean"]), lo=float(fair["delta_fpr"]["lo"]), hi=float(fair["delta_fpr"]["hi"])),
    )

    with open("out/bootstrap_ci.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved bootstrap CIs to out/bootstrap_ci.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model",  choices=["weighted","stacking","best"], default="stacking")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)) or {}
    main(cfg)
# when loading probabilities, select based on args.model
# e.g., from out/fusion_results.json or directly recompute:
# probs = get_probs(model=args.model)
