import torch, torch.nn as nn, torch.optim as optim, json, os
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from models.model_cnn import build_resnet18
from models.data import make_loaders
from models.losses import FocalLoss

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 50
BS = 64
LR = 1e-3
WD = 1e-4
ALPHA = 0.25
GAMMA = 2.0

os.makedirs("out", exist_ok=True)

def validate(model, loader):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            prob = torch.sigmoid(logits).squeeze(1).cpu().numpy().tolist()
            y_prob.extend(prob); y_true.extend(y.numpy().tolist())
    return roc_auc_score(y_true, y_prob)

def main():
    tr, va, te = make_loaders("data/train.csv", batch_size=BS, num_workers=4)
    model = build_resnet18(in_ch=3).to(DEVICE)
    crit = FocalLoss(alpha=ALPHA, gamma=GAMMA)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch = CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))

    best_auc, bad, patience = -1, 0, 10
    last_ckpts = []

    for ep in range(1, EPOCHS+1):
        model.train()
        pbar = tqdm(tr, desc=f"Epoch {ep}/{EPOCHS}")
        for x, y in pbar:
            x = x.to(DEVICE); y = y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=float(loss))

        sch.step()
        val_auc = validate(model, va)
        print(f"[val] AUC={val_auc:.4f}")

        # keep snapshot
        ckpt = f"out/ckpt_ep{ep:03d}.pt"
        torch.save(model.state_dict(), ckpt)
        last_ckpts.append(ckpt)
        if len(last_ckpts) > 5:
            os.remove(last_ckpts.pop(0))  # keep last 5 snapshots

        if val_auc > best_auc:
            torch.save(model.state_dict(), "out/best.pt")
            best_auc, bad = val_auc, 0
        else:
            bad += 1
            if bad >= patience: break

    # Evaluate best on test
    model.load_state_dict(torch.load("out/best.pt", map_location=DEVICE))
    test_auc = validate(model, te)
    print(f"[test] AUC={test_auc:.4f}")

    # Snapshot ensemble (average probs of last K=5 checkpoints)
    from statistics import mean
    import numpy as np
    model.eval()
    def predict_from_ckpt(ck):
        m = build_resnet18(in_ch=3).to(DEVICE); m.load_state_dict(torch.load(ck, map_location=DEVICE)); m.eval()
        probs = []
        with torch.no_grad():
            for x, _ in te:
                x = x.to(DEVICE); probs.append(torch.sigmoid(m(x)).squeeze(1).cpu().numpy())
        return np.concatenate(probs)
    if len(last_ckpts):
        preds = [predict_from_ckpt(ck) for ck in last_ckpts]
        import pandas as pd
        # rebuild test labels
        y_true = []
        for _, y in te:
            y_true.extend(y.numpy().tolist())
        y_true = np.array(y_true)

        ens = np.mean(np.vstack(preds), axis=0)
        ens_auc = roc_auc_score(y_true, ens)
        print(f"[test-ensemble(last {len(last_ckpts)})] AUC={ens_auc:.4f}")

        # save per-sample for bootstrap
        import json
        with open("out/test_probs.json","w") as f:
            json.dump({"y_true": y_true.tolist(), "prob_best": preds[-1].tolist(), "prob_ens": ens.tolist()}, f)

if __name__ == "__main__":
    main()

