import numpy as np

def load_synthetic(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, 4, size=n)
    tab = rng.normal(0, 1, size=n) + (groups*0.1)
    img = rng.normal(0, 1, size=n)
    seq = rng.normal(0, 1, size=n) + (groups==0)*0.2
    logit = 0.6*tab + 0.3*seq + 0.1*img + rng.normal(0, 1, size=n)
    y = (logit > 0.5).astype(int)
    sig = lambda z: 1/(1+np.exp(-z/1.5))
    return y, sig(tab), sig(seq), sig(img), groups
