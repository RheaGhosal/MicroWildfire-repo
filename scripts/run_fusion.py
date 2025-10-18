import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse, yaml, os, json
import numpy as np
from src.data import load_synthetic
from src.evaluation import evaluate_split, youden_threshold
from src.fusion import weighted_fusion, stacking_fusion
from src.metrics import compute_eo_spd_delta_fpr, auroc

# --- JSON sanitizer: convert numpy dtypes to native Python for JSON ---
def _to_native(obj):
    import numpy as np
    if isinstance(obj, dict):
        # ensure keys are strings (or native ints) and values are native
        out = {}
        for k, v in obj.items():
            # convert keys to str to be safe across numpy scalar keys
            if isinstance(k, (np.generic,)):
                k = k.item()
            k = str(k)
            out[k] = _to_native(v)
        return out
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    else:
        # numpy scalar → native
        if hasattr(obj, "item") and not isinstance(obj, (bytes, bytearray)):
            try:
                return obj.item()
            except Exception:
                pass
        return obj


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
    val_aurocs = {
        "tab": auroc(y[va], p_tab[va]),
        "seq": auroc(y[va], p_seq[va]),
        "img": auroc(y[va], p_img[va]),
    }
    best_name = max(val_aurocs, key=val_aurocs.get)
    p_best = {"tab": p_tab, "seq": p_seq, "img": p_img}[best_name]

    # Evaluate best single on TEST using the same threshold thr
    res_b = evaluate_split(y[te], p_best[te], groups=groups[te], threshold=thr)

    # ----- attach fairness summaries (EO/SPD/ΔFPR) at the same thr -----
    res_w["fairness"] = compute_eo_spd_delta_fpr(y[te], prob_w[te], groups[te], threshold=thr)
    res_s["fairness"] = compute_eo_spd_delta_fpr(y[te], prob_s,       groups[te], threshold=thr)
    res_b["fairness"] = compute_eo_spd_delta_fpr(y[te], p_best[te],    groups[te], threshold=thr)

    # ----- save everything -----
    out = {
        "threshold": float(thr),
        "val_aurocs": {k: float(v) for k, v in val_aurocs.items()},
        "best_single": best_name,
        "best_metrics": res_b,
        "weighted": res_w,
        "stacking": res_s,
    }
    out = _to_native(out)
    json.dump(out, open("out/fusion_results.json", "w"), indent=2)
    print("Saved fusion results to out/fusion_results.json")
    
    def proxy_auc(y, p): return ((p[y==1]).mean() - (p[y==0]).mean())
    choices = dict(tab=p_tab, seq=p_seq, img=p_img)
    best_name = max(choices, key=lambda k: proxy_auc(y[va], choices[k][va]))
    res_b = evaluate_split(y[te], choices[best_name][te], groups=groups[te], threshold=thr)
    json.dump(_to_native(dict(weighted=res_w, stacking=res_s, best_single=best_name, best_metrics=res_b)), open("out/fusion_results.json","w"), indent=2)
    print("Saved fusion results to out/fusion_results.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
