import argparse, yaml, os, json, numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def main(cfg):
    n = 5000
    y = np.random.randint(0,2,size=n)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=cfg["seed"])
    for train_idx, test_idx in sss.split(np.zeros_like(y), y):
        pass
    y_tr = y[train_idx]
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.1765, random_state=cfg["seed"])  # ~15% overall val
    for tr_idx, val_idx in sss2.split(np.zeros_like(y_tr), y_tr):
        pass
    out = dict(train_idx=train_idx[tr_idx].tolist(), val_idx=train_idx[val_idx].tolist(), test_idx=test_idx.tolist())
    os.makedirs("out", exist_ok=True)
    json.dump(out, open("out/splits.json","w"))
    print("Saved splits to out/splits.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
