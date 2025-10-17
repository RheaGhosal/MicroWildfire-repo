import argparse, yaml, os, json
import numpy as np
from src.data import load_synthetic
from src.evaluation import evaluate_split, youden_threshold
from src.fusion import weighted_fusion, stacking_fusion

def main(cfg):
    os.makedirs("out", exist_ok=True)
    y, p_tab, p_seq, p_img, groups = load_synthetic()
    splits = json.load(open("out/splits.json"))
    tr, va, te = np.array(splits["train_idx"]), np.array(splits["val_idx"]), np.array(splits["test_idx"])
    # weighted fusion
    prob_w = weighted_fusion([p_img, p_seq, p_tab], weights=cfg["fusion"]["weighted"]["weights"])
    thr = youden_threshold(y[va], prob_w[va])
    res_w = evaluate_split(y[te], prob_w[te], groups=groups[te], threshold=thr)
    # stacking fusion
    prob_s = stacking_fusion([p_img[tr], p_seq[tr], p_tab[tr]], y[tr], [p_img[te], p_seq[te], p_tab[te]])
    res_s = evaluate_split(y[te], prob_s, groups=groups[te], threshold=thr)
    # best single baseline (choose best on val by auroc proxy)
    # here we use a simple proxy since synthetic
    def proxy_auc(y, p): return ((p[y==1]).mean() - (p[y==0]).mean())
    choices = dict(tab=p_tab, seq=p_seq, img=p_img)
    best_name = max(choices, key=lambda k: proxy_auc(y[va], choices[k][va]))
    res_b = evaluate_split(y[te], choices[best_name][te], groups=groups[te], threshold=thr)
    json.dump(dict(weighted=res_w, stacking=res_s, best_single=best_name, best_metrics=res_b), open("out/fusion_results.json","w"), indent=2)
    print("Saved fusion results to out/fusion_results.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
