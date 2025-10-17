import numpy as np
from src.metrics import ece, auroc, brier, fairness_groups

def test_metrics_shapes():
    y = np.array([0,1,0,1])
    p = np.array([0.1,0.9,0.2,0.8])
    g = np.array([0,0,1,1])
    assert 0 <= ece(y,p) <= 1
    assert 0 <= auroc(y,p) <= 1
    assert brier(y,p) >= 0
    eo, spd, dfpr = fairness_groups(y,p,g)
    assert 0 <= eo <= 1
    assert 0 <= spd <= 1
    assert 0 <= dfpr <= 1
