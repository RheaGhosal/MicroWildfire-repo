import numpy as np
from sklearn.metrics import roc_curve
from .metrics import auroc, brier, ece, fairness_groups

def youden_threshold(y_true, y_prob):
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thr[np.argmax(j)])

def evaluate_split(y_true, y_prob, groups=None, threshold=None):
    if threshold is None:
        threshold = youden_threshold(y_true, y_prob)
    yhat = (y_prob >= threshold).astype(int)
    res = {}
    res["auroc"] = auroc(y_true, y_prob)
    res["brier"] = brier(y_true, y_prob)
    res["ece"] = ece(y_true, y_prob)
    if groups is not None:
        eo, spd, dfpr = fairness_groups(y_true, y_prob, groups)
        res.update(dict(eo=eo, spd=spd, delta_fpr=dfpr))
    res["threshold"] = threshold
    res["acc"] = float(np.mean(yhat == y_true))
    tp = np.sum((y_true==1) & (yhat==1))
    fp = np.sum((y_true==0) & (yhat==1))
    fn = np.sum((y_true==1) & (yhat==0))
    prec = tp / max(tp+fp, 1)
    rec = tp / max(tp+fn, 1)
    res["f1"] = 2*prec*rec / max(prec+rec, 1e-9)
    return res
