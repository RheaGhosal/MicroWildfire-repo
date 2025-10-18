import os, json, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

VAL_CSV  = "/storage/Wildfire/workspace/rev2_final/preds/VAL_preds.csv"
TEST_CSV = "/storage/Wildfire/workspace/rev2_final/preds/TEST_with_fusions.csv"
FUSION_JSON = "/storage/Wildfire/workspace/rev2_final/tables/fusion_summary.json"
OUT_TEX = "/storage/Wildfire/workspace/rev2_final/tables/table_acc_f1.tex"
MODEL_COL = "p_weighted"   # change to "p_stacking" if you prefer

def ensure_weighted(val_df, test_df):
    # If p_weighted exists, do nothing; else build from weights or grid on VAL
    if "p_weighted" in val_df.columns and "p_weighted" in test_df.columns:
        return val_df, test_df
    w = None
    if os.path.exists(FUSION_JSON):
        try:
            with open(FUSION_JSON, "r") as f:
                js = json.load(f)
            ww = js.get("val_best_weights", None)
            if ww and len(ww) == 3:
                w = tuple(float(x) for x in ww)
        except Exception:
            w = None
    if w is None:
        # grid-search weights on VAL for AUROC
        yv = val_df["y_true"].astype(int).values
        best_auc, best_w = -1.0, (1/3,1/3,1/3)
        for w1 in np.linspace(0,1,21):
            for w2 in np.linspace(0,1,21):
                if w1 + w2 <= 1:
                    w3 = 1 - w1 - w2
                    p = w1*val_df["p_img"] + w2*val_df["p_seq"] + w3*val_df["p_tab"]
                    try:
                        auc = roc_auc_score(yv, p)
                    except ValueError:
                        continue
                    if auc > best_auc:
                        best_auc, best_w = auc, (w1,w2,w3)
        w = best_w
    w1, w2, w3 = w
    val_df["p_weighted"]  = w1*val_df["p_img"] + w2*val_df["p_seq"] + w3*val_df["p_tab"]
    test_df["p_weighted"] = w1*test_df["p_img"] + w2*test_df["p_seq"] + w3*test_df["p_tab"]
    return val_df, test_df

def pick_thr_max_f1(y, p):
    grid = np.linspace(0.01,0.99,99)
    best = max(grid, key=lambda t: f1_score(y, (p>=t).astype(int)))
    return best

def boot_ci(fn, y, yhat, B=200, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    vals=[]
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(fn(y[idx], yhat[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return fn(y, yhat), float(lo), float(hi)

def main():
    VAL  = pd.read_csv(VAL_CSV)
    TEST = pd.read_csv(TEST_CSV)
    VAL, TEST = ensure_weighted(VAL, TEST)

    yv, pv = VAL["y_true"].astype(int).values, VAL[MODEL_COL].values
    thr = pick_thr_max_f1(yv, pv)

    yt, pt = TEST["y_true"].astype(int).values, TEST[MODEL_COL].values
    yhat = (pt >= thr).astype(int)

    acc, acc_lo, acc_hi = boot_ci(accuracy_score, yt, yhat, B=200, seed=1)
    f1,  f1_lo,  f1_hi  = boot_ci(f1_score, yt, yhat, B=200, seed=2)

    print(f"Threshold (VAL, max F1): {thr:.3f}")
    print(f"Accuracy: {acc:.3f} [{acc_lo:.3f}, {acc_hi:.3f}]")
    print(f"F1:       {f1:.3f} [{f1_lo:.3f}, {f1_hi:.3f}]")

    os.makedirs(os.path.dirname(OUT_TEX), exist_ok=True)
    with open(OUT_TEX, "w") as f:
        f.write("\\begin{table}[t]\\centering\\small\\setlength{\\tabcolsep}{5pt}\\renewcommand{\\arraystretch}{1.05}\n")
        f.write("\\caption{Accuracy and F1 with 95\\% bootstrap CIs ($B{=}200$) for the weighted fusion at the VAL-chosen threshold (max $F_1$).}\\label{tab:acc_f1}\n")
        f.write("\\begin{tabular}{lcc}\\toprule\nMetric & Value [95\\% CI] & Threshold\\\\\\midrule\n")
        f.write(f"Accuracy & {acc:.3f} [{acc_lo:.3f}, {acc_hi:.3f}] & {thr:.2f}\\\\\n")
        f.write(f"F$_1$     & {f1:.3f} [{f1_lo:.3f}, {f1_hi:.3f}] & {thr:.2f}\\\\\n")
        f.write("\\bottomrule\\end{tabular}\\end{table}\n")

if __name__ == "__main__":
    main()

