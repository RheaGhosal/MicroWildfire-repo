import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import numpy as np
from itertools import combinations

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

def _group_rates(y_true, y_hat, groups):
    """Compute per-group TPR, FPR, and positive rate."""
    groups = np.asarray(groups)
    y_true = np.asarray(y_true).astype(int)
    y_hat  = np.asarray(y_hat).astype(int)

    stats = {}
    for g in np.unique(groups):
        m = (groups == g)
        yt = y_true[m]; yh = y_hat[m]
        tp = np.sum((yt == 1) & (yh == 1))
        fn = np.sum((yt == 1) & (yh == 0))
        fp = np.sum((yt == 0) & (yh == 1))
        tn = np.sum((yt == 0) & (yh == 0))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        pos = np.mean(yh) if len(yh) else 0.0
        stats[g] = dict(tpr=float(tpr), fpr=float(fpr), pos=float(pos), n=int(len(yt)))
    return stats

def compute_eo_spd_delta_fpr(y_true, y_prob, groups, threshold=0.5):
    """
    Equalized Odds (EO): max over group pairs of max(|ΔTPR|, |ΔFPR|)
    SPD: max over group pairs of |Δ positive prediction rate|
    ΔFPR: mean absolute pairwise difference of FPR across groups
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    groups = np.asarray(groups)

    y_hat = (y_prob >= threshold).astype(int)
    stats = _group_rates(y_true, y_hat, groups)

    # pairwise deltas
    eo = 0.0
    spd_max = 0.0
    fpr_gaps = []
    uniq = list(stats.keys())
    for g, h in combinations(uniq, 2):
        d_tpr = abs(stats[g]["tpr"] - stats[h]["tpr"])
        d_fpr = abs(stats[g]["fpr"] - stats[h]["fpr"])
        d_pos = abs(stats[g]["pos"] - stats[h]["pos"])
        eo = max(eo, d_tpr, d_fpr)
        spd_max = max(spd_max, d_pos)
        fpr_gaps.append(d_fpr)

    delta_fpr = float(np.mean(fpr_gaps)) if fpr_gaps else 0.0
    return dict(
        eo=float(eo),
        spd=float(spd_max),
        delta_fpr=float(delta_fpr),
        per_group={str(k): v for k, v in stats.items()},
        threshold=float(threshold),
    )

def bootstrap_parity_cis(y_true, y_prob, groups, threshold=0.5, B=200, alpha=0.05, paired_ids=None, seed=42):
    """
    Bootstrap CIs for EO/SPD/ΔFPR. If paired_ids is provided, resample on unique IDs.
    Returns: dict with mean/lo/hi for each metric.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    groups = np.asarray(groups)

    def one(vals_idx):
        res = compute_eo_spd_delta_fpr(y_true[vals_idx], y_prob[vals_idx], groups[vals_idx], threshold)
        return res["eo"], res["spd"], res["delta_fpr"]

    vals = []
    if paired_ids is not None:
        ids = np.unique(paired_ids)
        for _ in range(B):
            res_ids = rng.choice(ids, size=len(ids), replace=True)
            m = np.isin(paired_ids, res_ids)
            vals.append(one(m))
    else:
        n = len(y_true)
        for _ in range(B):
            idx = rng.choice(np.arange(n), size=n, replace=True)
            vals.append(one(idx))

    arr = np.asarray(vals, dtype=float)  # shape (B, 3) -> [eo, spd, dfpr]
    out = {}
    for i, key in enumerate(["eo", "spd", "delta_fpr"]):
        mean = float(np.mean(arr[:, i]))
        lo   = float(np.quantile(arr[:, i], alpha/2))
        hi   = float(np.quantile(arr[:, i], 1 - alpha/2))
        out[key] = dict(mean=mean, lo=lo, hi=hi)
    return out

