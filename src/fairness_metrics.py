# fairness_metrics.py
import os, argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, log_loss

# ---------- Helpers ----------
EPS = 1e-6

def to_logit(p):
    p = np.clip(p, EPS, 1-EPS)
    return np.log(p/(1-p))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def ece(scores, y, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    idx  = np.digitize(scores, bins) - 1
    err = 0.0
    for b in range(n_bins):
        m = (idx == b)
        if not m.any(): continue
        err += m.mean() * abs(scores[m].mean() - y[m].mean())
    return float(err)

def pick_threshold_max_f1(y, p):
    thr_grid = np.linspace(0.01, 0.99, 99)
    best_thr, best_f1 = 0.5, -1
    for t in thr_grid:
        f1 = f1_score(y, (p >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    return best_thr, best_f1

def group_rates(y_true, y_pred, groups):
    # returns dict per group: TPR, FPR, PPR (positive prediction rate)
    out = {}
    gvals = np.unique(groups)
    for g in gvals:
        m = (groups == g)
        yt = y_true[m]; yp = y_pred[m]
        if len(yt) == 0:
            out[g] = {"TPR": np.nan, "FPR": np.nan, "PPR": np.nan}
            continue
        pos = (yt == 1); neg = (yt == 0)
        tpr = ( (yp[pos] == 1).sum() / max(1, pos.sum()) )
        fpr = ( (yp[neg] == 1).sum() / max(1, neg.sum()) )
        ppr = ( (yp == 1).sum() / len(yp) )
        out[g] = {"TPR": float(tpr), "FPR": float(fpr), "PPR": float(ppr)}
    return out

def eo_gap(rates):
    # max across groups of |TPR diff| and |FPR diff|, then max of the two maxima
    gs = list(rates.keys())
    max_tpr = 0.0
    max_fpr = 0.0
    for i in range(len(gs)):
        for j in range(i+1, len(gs)):
            g1, g2 = gs[i], gs[j]
            max_tpr = max(max_tpr, abs(rates[g1]["TPR"] - rates[g2]["TPR"]))
            max_fpr = max(max_fpr, abs(rates[g1]["FPR"] - rates[g2]["FPR"]))
    return float(max(max_tpr, max_fpr))

def spd_gap(rates):
    # max |PPR_g - PPR_g'|
    gs = list(rates.keys())
    m = 0.0
    for i in range(len(gs)):
        for j in range(i+1, len(gs)):
            m = max(m, abs(rates[gs[i]]["PPR"] - rates[gs[j]]["PPR"]))
    return float(m)

def delta_fpr_gap(rates):
    # max FPR - min FPR
    vals = [rates[g]["FPR"] for g in rates]
    return float(np.nanmax(vals) - np.nanmin(vals))

def bootstrap_ci(fn, y, p, groups, B=200, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    n = len(y)
    # stratify by group to preserve composition
    unique_groups = np.unique(groups)
    idx_by_g = {g: np.where(groups==g)[0] for g in unique_groups}
    sizes_by_g = {g: len(idx_by_g[g]) for g in unique_groups}
    for _ in range(B):
        res_idx = []
        for g in unique_groups:
            ids = idx_by_g[g]
            if len(ids)==0: continue
            res_idx.append(rng.choice(ids, size=len(ids), replace=True))
        res_idx = np.concatenate(res_idx)
        vals.append(fn(y[res_idx], p[res_idx], groups[res_idx]))
    vals = np.array(vals, dtype=float)
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def fairness_metrics(y, p, groups, thr):
    yhat = (p >= thr).astype(int)
    rates = group_rates(y, yhat, groups)
    eo  = eo_gap(rates)
    spd = spd_gap(rates)
    dF  = delta_fpr_gap(rates)
    return eo, spd, dF, rates

def temperature_scale_from_val(p_val, y_val, max_iter=500, lr=0.01):
    # Fit T to minimize NLL on VAL logits; we only have probabilities -> convert to logits.
    logits = to_logit(p_val)
    T = 1.0
    for _ in range(max_iter):
        # d/dT NLL(sigmoid(logits/T)) = (1/T^2) * sum( (pT - y) * logits ), where pT = sigmoid(logits/T)
        pT = sigmoid(logits / T)
        grad = np.mean((pT - y_val) * logits) / (T*T + EPS)
        T_new = T - lr * grad
        if T_new <= 0: T_new = T * 0.5
        if abs(T_new - T) < 1e-6:
            break
        T = T_new
    return float(T)

def apply_temperature(p, T):
    return sigmoid(to_logit(p) / T)

def format_ci(mean, lo, hi, digits=3):
    return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_csv",  default="/storage/Wildfire/workspace/rev2_final/preds/VAL_preds.csv")
    ap.add_argument("--test_csv", default="/storage/Wildfire/workspace/rev2_final/preds/TEST_with_fusions.csv")
    ap.add_argument("--model_col", default="p_weighted", choices=["p_img","p_seq","p_tab","p_weighted","p_stacking"])
    ap.add_argument("--group_col", default="climate_zone")
    ap.add_argument("--B", type=int, default=200, help="bootstrap repetitions")
    ap.add_argument("--out_dir", default="/storage/Wildfire/workspace/rev2_final/tables")
    ap.add_argument("--fusion_json", default="/storage/Wildfire/workspace/rev2_final/tables/fusion_summary.json")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    VAL  = pd.read_csv(args.val_csv)
    TEST = pd.read_csv(args.test_csv)

    # ---- ensure fusion columns exist in both VAL and TEST if requested ----
    want_weighted = (args.model_col == "p_weighted")
    want_stacking = (args.model_col == "p_stacking")

    if want_weighted and "p_weighted" not in VAL.columns:
        # try to get weights from fusion_summary.json (preferred)
        w = None
        if os.path.exists(args.fusion_json):
            try:
                with open(args.fusion_json, "r") as f:
                    js = json.load(f)
                w = js.get("val_best_weights", None)
                if w is not None and len(w) == 3:
                    w1, w2, w3 = float(w[0]), float(w[1]), float(w[2])
                else:
                    w = None
            except Exception:
                w = None
        # fallback: small grid-search on VAL
        if w is None:
            best_auc, best_w = -1.0, (1/3,1/3,1/3)
            yv = VAL["y_true"].astype(int).values
            for w1 in np.linspace(0,1,21):
                for w2 in np.linspace(0,1,21):
                    if w1 + w2 <= 1:
                        w3 = 1 - w1 - w2
                        p = w1*VAL["p_img"] + w2*VAL["p_seq"] + w3*VAL["p_tab"]
                        auc = roc_auc_score(yv, p)
                        if auc > best_auc:
                            best_auc, best_w = auc, (w1,w2,w3)
            w1,w2,w3 = best_w
        # create fusion columns in both VAL and TEST
        VAL["p_weighted"]  = w1*VAL["p_img"] + w2*VAL["p_seq"] + w3*VAL["p_tab"]
        TEST["p_weighted"] = w1*TEST["p_img"] + w2*TEST["p_seq"] + w3*TEST["p_tab"]

    if want_stacking and "p_stacking" not in VAL.columns:
        # fit logistic stacker on VAL, apply to both
        from sklearn.linear_model import LogisticRegression
        yv = VAL["y_true"].astype(int).values
        Xv = VAL[["p_img","p_seq","p_tab"]].values
        Xt = TEST[["p_img","p_seq","p_tab"]].values
        lr = LogisticRegression(max_iter=1000)
        lr.fit(Xv, yv)
        VAL["p_stacking"]  = lr.predict_proba(Xv)[:,1]
        TEST["p_stacking"] = lr.predict_proba(Xt)[:,1]

    # pick probabilities to use
    if args.model_col not in VAL.columns or args.model_col not in TEST.columns:
        raise KeyError(f"model_col '{args.model_col}' not present after fusion construction. "
                       f"Have columns VAL={list(VAL.columns)}, TEST={list(TEST.columns)}")

    # 1) choose threshold on VAL to maximize F1 (pre)
    yv, pv = VAL["y_true"].astype(int).values, VAL[args.model_col].values
    thr_pre, f1_pre = pick_threshold_max_f1(yv, pv)

    # 2) temperature scaling on VAL, apply to TEST
    T = temperature_scale_from_val(pv, yv, max_iter=800, lr=0.01)
    print(f"[INFO] chosen model={args.model_col}, thr_pre(max F1 on VAL)={thr_pre:.3f}, temp T={T:.3f}")

    # 3) fairness on TEST, pre and post
    yt  = TEST["y_true"].astype(int).values
    pt  = TEST[args.model_col].values
    gt  = TEST[args.group_col].astype(str).values

    eo_pre, spd_pre, dF_pre, rates_pre = fairness_metrics(yt, pt, gt, thr_pre)

    pt_post = apply_temperature(pt, T)
    thr_post, f1_post = pick_threshold_max_f1(yv, apply_temperature(pv, T))
    eo_post, spd_post, dF_post, rates_post = fairness_metrics(yt, pt_post, gt, thr_post)

    # 4) bootstrap CIs
    def eo_fn(y, p, g): return fairness_metrics(y, p, g, thr_pre)[0]
    def spd_fn(y, p, g): return fairness_metrics(y, p, g, thr_pre)[1]
    def dF_fn(y, p, g):  return fairness_metrics(y, p, g, thr_pre)[2]
    eo_m, eo_lo, eo_hi   = bootstrap_ci(eo_fn, yt, pt, gt, B=args.B)
    spd_m, spd_lo, spd_hi= bootstrap_ci(spd_fn, yt, pt, gt, B=args.B)
    dF_m, dF_lo, dF_hi  = bootstrap_ci(dF_fn, yt, pt, gt, B=args.B)

    def eo_fn_post(y, p, g): return fairness_metrics(y, apply_temperature(p, T), g, thr_post)[0]
    def spd_fn_post(y, p, g):return fairness_metrics(y, apply_temperature(p, T), g, thr_post)[1]
    def dF_fn_post(y, p, g): return fairness_metrics(y, apply_temperature(p, T), g, thr_post)[2]
    eo_m2, eo_lo2, eo_hi2   = bootstrap_ci(eo_fn_post, yt, pt, gt, B=args.B)
    spd_m2, spd_lo2, spd_hi2= bootstrap_ci(spd_fn_post, yt, pt, gt, B=args.B)
    dF_m2, dF_lo2, dF_hi2  = bootstrap_ci(dF_fn_post, yt, pt, gt, B=args.B)

    # 5) ECE per-group
    groups_sorted = sorted(np.unique(gt))
    ece_rows = []
    for g in groups_sorted:
        m = (gt == g)
        e_pre  = ece(pt[m], yt[m], 10)
        e_post = ece(pt_post[m], yt[m], 10)
        ece_rows.append([g, e_pre, e_post])

    # 6) write LaTeX + summary
    fairness_tex = os.path.join(args.out_dir, "table_fairness.tex")
    with open(fairness_tex, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Fairness metrics with 95\\% bootstrap CIs ($B{=}" + str(args.B) + "$). Lower is better.}\n")
        f.write("\\label{tab:fairness}\n\\small\n\\setlength{\\tabcolsep}{3.5pt}\n\\renewcommand{\\arraystretch}{1.08}\n")
        f.write("\\begin{tabular}{lccc}\n\\toprule\n")
        f.write("\\textbf{Metric} & \\textbf{Pre} & \\textbf{Post (calibrated)} & \\textbf{Absolute $\\Delta$} \\\\\n\\midrule\n")
        f.write(f"EO  & {format_ci(eo_m, eo_lo, eo_hi)} & {format_ci(eo_m2, eo_lo2, eo_hi2)} & {eo_m2-eo_m:+.3f} \\\\\n")
        f.write(f"SPD & {format_ci(spd_m, spd_lo, spd_hi)} & {format_ci(spd_m2, spd_lo2, spd_hi2)} & {spd_m2-spd_m:+.3f} \\\\\n")
        f.write(f"$\\Delta$FPR & {format_ci(dF_m, dF_lo, dF_hi)} & {format_ci(dF_m2, dF_lo2, dF_hi2)} & {dF_m2-dF_m:+.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    ece_tex = os.path.join(args.out_dir, "table_ece_groups.tex")
    with open(ece_tex, "w") as f:
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{Subgroup calibration (ECE, lower is better).}\n")
        f.write("\\label{tab:ece_groups}\n\\small\n\\setlength{\\tabcolsep}{6pt}\n\\renewcommand{\\arraystretch}{1.08}\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("\\textbf{Group} & \\textbf{ECE (pre)} & \\textbf{ECE (post)} \\\\\n\\midrule\n")
        for g, e_pre, e_post in ece_rows:
            f.write(f"{g} & {e_pre:.3f} & \\textbf{{{e_post:.3f}}} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    summary_txt = os.path.join(args.out_dir, "fairness_summary.txt")
    with open(summary_txt, "w") as f:
        f.write("Fairness summary (auto-generated)\n")
        f.write(f"EO:  pre {format_ci(eo_m, eo_lo, eo_hi)}  -> post {format_ci(eo_m2, eo_lo2, eo_hi2)} (Δ {eo_m2-eo_m:+.3f})\n")
        f.write(f"SPD: pre {format_ci(spd_m, spd_lo, spd_hi)} -> post {format_ci(spd_m2, spd_lo2, spd_hi2)} (Δ {spd_m2-spd_m:+.3f})\n")
        f.write(f"ΔFPR: pre {format_ci(dF_m, dF_lo, dF_hi)} -> post {format_ci(dF_m2, dF_lo2, dF_hi2)} (Δ {dF_m2-dF_m:+.3f})\n")
        f.write(f"Model={args.model_col}\n")

    print("Wrote:")
    print("  LaTeX fairness table :", fairness_tex)
    print("  LaTeX ECE table      :", ece_tex)
    print("  Summary               :", summary_txt)

if __name__ == "__main__":
    main()

