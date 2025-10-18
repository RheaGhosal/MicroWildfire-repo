# run_pipeline.py  (resumable)
# Train/load CNN, LSTM, RF -> predict -> fuse -> stats with CIs and p-values.
# Resumes safely after reboot using checkpoints + a progress.json ledger.

import os, json, csv, random, argparse, time, tempfile, shutil
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import joblib

# ------------- Utils -------------
def atomic_write_bytes(path, data_bytes):
    """Write bytes atomically to avoid half-written files on crashes."""
    d = os.path.dirname(path); os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_", suffix=os.path.basename(path))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data_bytes)
        os.replace(tmp, path)
    finally:
        try: os.remove(tmp)
        except: pass

def atomic_write_text(path, text):
    atomic_write_bytes(path, text.encode("utf-8"))

def atomic_write_json(path, obj):
    atomic_write_text(path, json.dumps(obj, indent=2))

# ------------- Args -------------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/storage/Wildfire/workspace/data")
    ap.add_argument("--out-dir",   default="/storage/Wildfire/workspace/rev2_final")
    ap.add_argument("--ckpt-dir",  default="/storage/Wildfire/workspace/ckpts")
    ap.add_argument("--val-index", default=None, help="override path to val_index.csv")
    ap.add_argument("--test-index",default=None, help="override path to test_index.csv")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seq-hid", type=int, default=64)
    ap.add_argument("--seq-layers", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=2000, help="B for CI bootstraps")
    ap.add_argument("--resume", action="store_true", help="resume training if checkpoints exist")
    ap.add_argument("--retrain-cnn", action="store_true")
    ap.add_argument("--retrain-lstm", action="store_true")
    ap.add_argument("--retrain-rf", action="store_true")
    ap.add_argument("--recompute-preds", action="store_true")
    ap.add_argument("--recompute-stats", action="store_true")
    return ap.parse_args()

args = get_args()

DATA_ROOT = args.data_root
VAL_INDEX = args.val_index or f"{DATA_ROOT}/val_index.csv"
TEST_INDEX= args.test_index or f"{DATA_ROOT}/test_index.csv"

OUT_DIR   = args.out_dir
CKPT_DIR  = args.ckpt_dir
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(f"{OUT_DIR}/preds", exist_ok=True)
os.makedirs(f"{OUT_DIR}/tables", exist_ok=True)

VAL_OUT   = f"{OUT_DIR}/preds/VAL_preds.csv"
TEST_OUT  = f"{OUT_DIR}/preds/TEST_preds.csv"
TEST_FUSE = f"{OUT_DIR}/preds/TEST_with_fusions.csv"
TABLE_OUT = f"{OUT_DIR}/tables/Table_F1_fusion.csv"
JSON_OUT  = f"{OUT_DIR}/tables/fusion_summary.json"
LEDGER    = f"{OUT_DIR}/progress.json"

IMG_SIZE  = args.img_size
SEQ_HID   = args.seq_hid
SEQ_LAYERS= args.seq_layers
BATCH     = args.batch
EPOCHS    = args.epochs
LR        = args.lr
SEED      = args.seed
BOOT_B    = args.bootstrap

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------- Progress ledger -------------
def load_ledger():
    if os.path.exists(LEDGER):
        with open(LEDGER, "r") as f: return json.load(f)
    return {
        "cnn": {"done": False, "last_epoch": 0},
        "lstm":{"done": False, "last_epoch": 0},
        "rf":  {"done": False},
        "val_preds": False,
        "test_preds": False,
        "fusion": False
    }

def save_ledger(L):
    atomic_write_json(LEDGER, L)

L = load_ledger()

# ------------- Data -------------
class MultiModalDS(Dataset):
    def __init__(self, index_csv, img_size=224):
        self.df = pd.read_csv(index_csv)
        self.tfm = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
        try:
            arr = np.load(self.df["seq_path"].iloc[0])
            self.seq_dim = 1 if arr.ndim == 1 else arr.shape[-1]
        except Exception:
            self.seq_dim = 1
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        x_img = self.tfm(Image.open(r["image_path"]).convert("RGB"))
        seq = np.load(r["seq_path"])
        if seq.ndim == 1: seq = seq[:, None]
        x_seq = torch.tensor(seq, dtype=torch.float32)
        x_tab = torch.tensor(np.load(r["feat_path"]), dtype=torch.float32)
        y = torch.tensor(int(r["label"]), dtype=torch.long)
        meta = {k: r.get(k, "NA") for k in ["id","region","climate_zone","county_fips"]}
        return x_img, x_seq, x_tab, y, meta

val_ds  = MultiModalDS(VAL_INDEX, IMG_SIZE)
test_ds = MultiModalDS(TEST_INDEX, IMG_SIZE)
SEQ_DIM = val_ds.seq_dim

val_loader  = DataLoader(val_ds, batch_size=BATCH, shuffle=True,  num_workers=0, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH, shuffle=False, num_workers=0, pin_memory=True)

# ------------- Models -------------
class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        self.m = m
    def forward(self, x): return self.m(x).squeeze(1)

class TinyLSTM(nn.Module):
    def __init__(self, in_dim=1, hid=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid, num_layers=layers, batch_first=True)
        self.fc   = nn.Linear(hid, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        return self.fc(h).squeeze(1)

# ------------- Checkpoints -------------
CNN_CKPT = f"{CKPT_DIR}/cnn_state.pt"    # contains model, optimizer, last_epoch
LSTM_CKPT= f"{CKPT_DIR}/lstm_state.pt"
RF_CKPT  = f"{CKPT_DIR}/rf.joblib"

def save_ckpt(path, state_dict):
    b = json.dumps({"_marker":"pt-state"}, indent=0).encode()  # small marker to ensure non-empty file
    atomic_write_bytes(path + ".marker", b)  # side file to ensure fs sync
    torch.save(state_dict, path + ".tmp")
    os.replace(path + ".tmp", path)

def load_ckpt(path, map_location=None):
    return torch.load(path, map_location=map_location)

# ------------- Train / Resume -------------
def train_or_resume_cnn():
    start_epoch = 0
    model = TinyCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()

    if (args.resume or not args.retrain_cnn) and os.path.exists(CNN_CKPT):
        try:
            state = load_ckpt(CNN_CKPT, map_location=device)
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            start_epoch = state.get("epoch", 0)
            print(f"[CNN] Resuming from epoch {start_epoch}/{EPOCHS}")
        except Exception as e:
            print("[CNN] Failed to resume, retraining from scratch:", e)

    if L["cnn"]["done"] and not args.retrain_cnn:
        model.eval()
        print("[CNN] Already marked done; skipping training.")
        return model

    # train remaining epochs
    for ep in range(start_epoch, EPOCHS):
        model.train()
        tot=0.0
        for x_img, _, _, y, _ in val_loader:
            x_img, y = x_img.to(device), y.float().to(device)
            opt.zero_grad()
            logit = model(x_img)
            loss = lossf(logit, y)
            loss.backward(); opt.step()
            tot += loss.item() * y.size(0)
        print(f"[CNN] epoch {ep+1}/{EPOCHS} loss={tot/len(val_ds):.4f}")
        # checkpoint after each epoch
        save_ckpt(CNN_CKPT, {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep+1})
        L["cnn"]["last_epoch"] = ep+1; save_ledger(L)

    L["cnn"]["done"] = True; save_ledger(L)
    model.eval()
    return model

def train_or_resume_lstm():
    start_epoch = 0
    model = TinyLSTM(in_dim=SEQ_DIM, hid=SEQ_HID, layers=SEQ_LAYERS).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()

    if (args.resume or not args.retrain_lstm) and os.path.exists(LSTM_CKPT):
        try:
            state = load_ckpt(LSTM_CKPT, map_location=device)
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            start_epoch = state.get("epoch", 0)
            print(f"[LSTM] Resuming from epoch {start_epoch}/{EPOCHS}")
        except Exception as e:
            print("[LSTM] Failed to resume, retraining from scratch:", e)

    if L["lstm"]["done"] and not args.retrain_lstm:
        model.eval()
        print("[LSTM] Already marked done; skipping training.")
        return model

    for ep in range(start_epoch, EPOCHS):
        model.train()
        tot=0.0
        for _, x_seq, _, y, _ in val_loader:
            x_seq, y = x_seq.to(device), y.float().to(device)
            opt.zero_grad()
            logit = model(x_seq)
            loss = lossf(logit, y)
            loss.backward(); opt.step()
            tot += loss.item() * y.size(0)
        print(f"[LSTM] epoch {ep+1}/{EPOCHS} loss={tot/len(val_ds):.4f}")
        save_ckpt(LSTM_CKPT, {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep+1})
        L["lstm"]["last_epoch"] = ep+1; save_ledger(L)

    L["lstm"]["done"] = True; save_ledger(L)
    model.eval()
    return model

def train_or_resume_rf():
    if (not args.retrain_rf) and L["rf"]["done"] and os.path.exists(RF_CKPT):
        print("[RF] Already trained; loading.")
        return joblib.load(RF_CKPT)

    # If joblib exists and we're not forcing retrain, just load
    if (not args.retrain_rf) and os.path.exists(RF_CKPT):
        try:
            print("[RF] Loading existing checkpoint.")
            return joblib.load(RF_CKPT)
        except Exception as e:
            print("[RF] Failed to load existing RF, retraining:", e)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    X, y = [], []
    for _, _, x_tab, yy, _ in val_loader:
        X.append(x_tab.numpy()); y.append(yy.numpy())
    X = np.concatenate(X, 0); y = np.concatenate(y, 0).astype(int)
    rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1).fit(X, y)
    joblib.dump(rf, RF_CKPT)
    try:
        auc = roc_auc_score(y, rf.predict_proba(X)[:,1])
        print(f"[RF] trained AUC(val)={auc:.3f}")
    except Exception:
        print("[RF] trained.")
    L["rf"]["done"] = True; save_ledger(L)
    return rf

cnn  = train_or_resume_cnn()
lstm = train_or_resume_lstm()
rf   = train_or_resume_rf()

# ------------- Predict (resumable) -------------
@torch.no_grad()
def predict_split(loader, out_csv):
    rows=[]
    for x_img, x_seq, x_tab, y, meta in loader:
        p_img = torch.sigmoid(cnn(x_img.to(device))).cpu().numpy()
        p_seq = torch.sigmoid(lstm(x_seq.to(device))).cpu().numpy()
        p_tab = rf.predict_proba(x_tab.numpy())[:,1]
        for i in range(y.size(0)):
            rows.append([
                meta["id"], int(y[i].item()),
                float(p_img[i]), float(p_seq[i]), float(p_tab[i]),
                meta.get("region","NA"), meta.get("climate_zone","NA"), meta.get("county_fips","NA")
            ])
    df = pd.DataFrame(rows, columns=["id","y_true","p_img","p_seq","p_tab","region","climate_zone","county_fips"])
    atomic_write_text(out_csv, df.to_csv(index=False))

if (not os.path.exists(VAL_OUT)) or args.recompute_preds or (not L["val_preds"]):
    print("[PRED] Generating VAL predictions …")
    predict_split(val_loader, VAL_OUT)
    L["val_preds"] = True; save_ledger(L)
else:
    print("[PRED] VAL predictions exist; skipping.")

if (not os.path.exists(TEST_OUT)) or args.recompute_preds or (not L["test_preds"]):
    print("[PRED] Generating TEST predictions …")
    predict_split(test_loader, TEST_OUT)
    L["test_preds"] = True; save_ledger(L)
else:
    print("[PRED] TEST predictions exist; skipping.")

# ------------- Fusion + Stats (resumable) -------------
rng = np.random.default_rng(SEED)

def ece(scores, y, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(scores, bins) - 1
    e = 0.0
    for b in range(n_bins):
        m = (idx == b)
        if not m.any(): continue
        e += m.mean() * abs(scores[m].mean() - y[m].mean())
    return float(e)

def boot_ci(y, p, metric="auc", B=2000):
    vals=[]
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if metric == "auc":
            vals.append(roc_auc_score(y[idx], p[idx]))
        elif metric == "brier":
            vals.append(brier_score_loss(y[idx], p[idx]))
        elif metric == "ece":
            vals.append(ece(p[idx], y[idx], 10))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)

def delta_auc_p(y, a, b, B=2000):
    vals=[]
    n = len(y)
    for _ in range(B):
        idx = rng.integers(0, n, n)
        vals.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    vals = np.array(vals)
    pval = 2 * min((vals <= 0).mean(), (vals >= 0).mean())
    return float(np.mean(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(pval)

def fmt(m): return f"{m[0]:.3f} [{m[1]:.3f},{m[2]:.3f}]"

def do_fusion_and_stats():
    VAL  = pd.read_csv(VAL_OUT)
    TEST = pd.read_csv(TEST_OUT)
    yv, yt = VAL.y_true.values, TEST.y_true.values

    # grid-search weights on VAL
    best_auc, best_W = -1.0, (1/3, 1/3, 1/3)
    for w1 in np.linspace(0,1,21):
        for w2 in np.linspace(0,1,21):
            if w1 + w2 <= 1:
                w3 = 1 - w1 - w2
                auc = roc_auc_score(yv, w1*VAL.p_img + w2*VAL.p_seq + w3*VAL.p_tab)
                if auc > best_auc:
                    best_auc, best_W = auc, (w1, w2, w3)
    w1, w2, w3 = best_W
    p_weighted = w1*TEST.p_img + w2*TEST.p_seq + w3*TEST.p_tab

    # stacking (logistic) on VAL
    stacker = LogisticRegression(max_iter=1000)
    stacker.fit(VAL[["p_img","p_seq","p_tab"]], yv)
    p_stacking = stacker.predict_proba(TEST[["p_img","p_seq","p_tab"]])[:,1]

    # metrics
    singles = {
      "img": {"auc": boot_ci(yt, TEST.p_img.values, "auc", B=BOOT_B),
              "brier": boot_ci(yt, TEST.p_img.values, "brier", B=BOOT_B),
              "ece": boot_ci(yt, TEST.p_img.values, "ece", B=BOOT_B)},
      "seq": {"auc": boot_ci(yt, TEST.p_seq.values, "auc", B=BOOT_B),
              "brier": boot_ci(yt, TEST.p_seq.values, "brier", B=BOOT_B),
              "ece": boot_ci(yt, TEST.p_seq.values, "ece", B=BOOT_B)},
      "tab": {"auc": boot_ci(yt, TEST.p_tab.values, "auc", B=BOOT_B),
              "brier": boot_ci(yt, TEST.p_tab.values, "brier", B=BOOT_B),
              "ece": boot_ci(yt, TEST.p_tab.values, "ece", B=BOOT_B)},
    }
    fusion = {
      "weighted": {"auc": boot_ci(yt, p_weighted, "auc", B=BOOT_B),
                   "brier": boot_ci(yt, p_weighted, "brier", B=BOOT_B),
                   "ece": boot_ci(yt, p_weighted, "ece", B=BOOT_B)},
      "stacking": {"auc": boot_ci(yt, p_stacking, "auc", B=BOOT_B),
                   "brier": boot_ci(yt, p_stacking, "brier", B=BOOT_B),
                   "ece": boot_ci(yt, p_stacking, "ece", B=BOOT_B)},
    }
    best_single = max(singles, key=lambda k: singles[k]["auc"][0])
    d_w = delta_auc_p(yt, p_weighted, TEST[f"p_{best_single}"].values, B=BOOT_B)
    d_s = delta_auc_p(yt, p_stacking, TEST[f"p_{best_single}"].values, B=BOOT_B)

    # save table
    rows = [
        ["CNN (imagery)",          fmt(singles["img"]["auc"]), fmt(singles["img"]["brier"]), fmt(singles["img"]["ece"])],
        ["LSTM (sequence)",        fmt(singles["seq"]["auc"]), fmt(singles["seq"]["brier"]), fmt(singles["seq"]["ece"])],
        ["RF (tabular)",           fmt(singles["tab"]["auc"]), fmt(singles["tab"]["brier"]), fmt(singles["tab"]["ece"])],
        ["Fusion: weighted avg",   fmt(fusion["weighted"]["auc"]), fmt(fusion["weighted"]["brier"]), fmt(fusion["weighted"]["ece"])],
        ["Fusion: stacking (LR)",  fmt(fusion["stacking"]["auc"]), fmt(fusion["stacking"]["brier"]), fmt(fusion["stacking"]["ece"])],
    ]
    atomic_write_text(TABLE_OUT, pd.DataFrame(rows, columns=["Model","AUROC (95% CI)","Brier (95% CI)","ECE (95% CI)"]).to_csv(index=False))

    TEST2 = TEST.copy()
    TEST2["p_weighted"] = p_weighted
    TEST2["p_stacking"] = p_stacking
    atomic_write_text(TEST_FUSE, TEST2.to_csv(index=False))

    summary = {
      "val_best_weights": list((w1, w2, w3)),
      "best_single": best_single,
      "singles_auc": {k: list(v["auc"]) for k,v in singles.items()},
      "fusion_auc":  {k: list(v["auc"]) for k,v in fusion.items()},
      "fusion_brier":{k: list(v["brier"]) for k,v in fusion.items()},
      "fusion_ece":  {k: list(v["ece"]) for k,v in fusion.items()},
      "delta_auc_weighted": {"mean": d_w[0], "ci":[d_w[1], d_w[2]], "p": d_w[3]},
      "delta_auc_stacking": {"mean": d_s[0], "ci":[d_s[1], d_s[2]], "p": d_s[3]},
      "bootstrap_B": BOOT_B
    }
    atomic_write_json(JSON_OUT, summary)

if (not os.path.exists(JSON_OUT)) or args.recompute_stats or (not L["fusion"]):
    print("[FUSION] Computing fusion metrics …")
    do_fusion_and_stats()
    L["fusion"] = True; save_ledger(L)
else:
    print("[FUSION] Fusion/stats exist; skipping.")

print("✅ DONE")
print("  Outputs:")
print("   -", VAL_OUT)
print("   -", TEST_OUT)
print("   -", TEST_FUSE)
print("   -", TABLE_OUT)
print("   -", JSON_OUT)
print("  Ledger:", LEDGER)
print("  Device:", "cuda" if torch.cuda.is_available() else "cpu")

