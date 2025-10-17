import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve

def auroc(y_true, y_prob):
    return roc_auc_score(y_true, y_prob)

def brier(y_true, y_prob):
    return brier_score_loss(y_true, y_prob)

def ece(y_true, y_prob, n_bins=15):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    # simple ECE estimate
    return float(np.sum(np.abs(prob_true - prob_pred)) / len(prob_true))

def fairness_groups(y_true, y_prob, groups):
    # Compute EO, SPD, ΔFPR across groups, using thresholded predictions (0.5 by default)
    unique = np.unique(groups)
    stats = {}
    yhat = (y_prob >= 0.5).astype(int)
    for g in unique:
        idx = (groups == g)
        y_g = y_true[idx]; yh_g = yhat[idx]
        tp = np.sum((y_g == 1) & (yh_g == 1))
        fn = np.sum((y_g == 1) & (yh_g == 0))
        fp = np.sum((y_g == 0) & (yh_g == 1))
        tn = np.sum((y_g == 0) & (yh_g == 0))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        pos = np.mean(yh_g)
        stats[g] = dict(tpr=tpr, fpr=fpr, pos=pos)
    # pairwise
    pairs = [(unique[i], unique[j]) for i in range(len(unique)) for j in range(i+1, len(unique))]
    eo = 0.0; spd_max = 0.0; fpr_diffs = []
    for g,h in pairs:
        eo = max(eo, abs(stats[g]["tpr"]-stats[h]["tpr"]), abs(stats[g]["fpr"]-stats[h]["fpr"]))
        spd_max = max(spd_max, abs(stats[g]["pos"]-stats[h]["pos"]))
        fpr_diffs.append(abs(stats[g]["fpr"]-stats[h]["fpr"]))
    delta_fpr = float(np.mean(fpr_diffs)) if fpr_diffs else 0.0
    return eo, spd_max, delta_fpr

def bootstrap_ci(metric_fn, y_true, y_prob, groups=None, B=200, paired_ids=None, alpha=0.05):
    # Generic bootstrap CI; if paired_ids provided, resample on unique ids (paired bootstrap)
    rng = np.random.default_rng(42)
    vals = []
    if paired_ids is not None:
        ids = np.unique(paired_ids)
        for _ in range(B):
            res_ids = rng.choice(ids, size=len(ids), replace=True)
            mask = np.isin(paired_ids, res_ids)
            if groups is None:
                vals.append(metric_fn(y_true[mask], y_prob[mask]))
            else:
                vals.append(metric_fn(y_true[mask], y_prob[mask], groups[mask]))
    else:
        n = len(y_true)
        for _ in range(B):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            if groups is None:
                vals.append(metric_fn(y_true[idx], y_prob[idx]))
            else:
                vals.append(metric_fn(y_true[idx], y_prob[idx], groups[idx]))
    vals = np.array(vals, dtype=float)
    lo = np.quantile(vals, alpha/2); hi = np.quantile(vals, 1 - alpha/2)
    return float(np.mean(vals)), float(lo), float(hi)
