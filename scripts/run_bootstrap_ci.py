import argparse, yaml, os, json
import numpy as np
from src.data import load_synthetic
from src.metrics import bootstrap_ci, auroc, brier, ece, fairness_groups

def main(cfg):
    os.makedirs("out", exist_ok=True)
    y, p_tab, p_seq, p_img, groups = load_synthetic()
    B = cfg["bootstrap"]["B"]
    mean_auc, lo_auc, hi_auc = bootstrap_ci(auroc, y, p_tab, B=B)
    mean_brier, lo_brier, hi_brier = bootstrap_ci(brier, y, p_tab, B=B)
    mean_ece, lo_ece, hi_ece = bootstrap_ci(ece, y, p_tab, B=B)
    def fairness_metric(y_true, y_prob, groups):
        eo, spd, dfpr = fairness_groups(y_true, y_prob, groups)
        return eo
    mean_eo, lo_eo, hi_eo = bootstrap_ci(fairness_metric, y, p_tab, groups=groups, B=B)
    json.dump(dict(auroc=[mean_auc, lo_auc, hi_auc],
                   brier=[mean_brier, lo_brier, hi_brier],
                   ece=[mean_ece, lo_ece, hi_ece],
                   eo=[mean_eo, lo_eo, hi_eo]), open("out/bootstrap_ci.json","w"), indent=2)
    print("Saved bootstrap CIs to out/bootstrap_ci.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
