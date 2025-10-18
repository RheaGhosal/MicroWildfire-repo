# make_synth_data.py
import os, random, numpy as np, pandas as pd
from PIL import Image, ImageDraw

random.seed(42); np.random.seed(42)

ROOT = "/storage/Wildfire/workspace/data"
N_VAL  = 600
N_TEST = 600

def mk_dirs():
    for split in ["val","test"]:
        for sub in ["images","seq","tab"]:
            os.makedirs(f"{ROOT}/{split}/{sub}", exist_ok=True)

def draw_img(val_signal, w=64, h=64):
    # simple image: brighter diagonal if signal high
    img = Image.new("RGB", (w,h), (int(80+120*val_signal),)*3)
    d = ImageDraw.Draw(img)
    for i in range(0, w, 8):
        c = int(60 + 180*val_signal)
        d.line((0,i,w,i), fill=(c, c//2, 255-c//2))
    return img

def make_split(split, n):
    rows=[]
    regions = ["NW","NE","SW","SE"]
    climates = ["A","B","C","D"]  # pretend Köppen classes
    for i in range(n):
        _id = f"{i:04d}"
        # latent true drivers
        s_img = np.clip(np.random.beta(2,2), 0, 1)         # image signal
        s_seq = np.clip(np.random.normal(0.5, 0.2), 0, 1)  # seq signal
        s_tab = np.clip(np.random.uniform(), 0, 1)         # tab signal

        # combine with region-specific weights to encourage fairness analysis
        reg = random.choice(regions)
        cz  = random.choice(climates)
        fips = f"{random.randint(10001, 99999)}"
        w_img = 0.45 if reg in ["NW","SE"] else 0.25
        w_seq = 0.35 if reg in ["NE","SE"] else 0.25
        w_tab = 1.0 - w_img - w_seq

        logit = 2.2*(w_img*s_img + w_seq*s_seq + w_tab*s_tab - 0.55)
        prob  = 1/(1+np.exp(-logit))
        y     = int(np.random.rand() < prob)

        # create modalities
        img = draw_img(s_img); img_path = f"{ROOT}/{split}/images/{_id}.jpg"; img.save(img_path)
        seq = (s_seq + 0.15*np.random.randn(20)).astype("float32")
        np.save(f"{ROOT}/{split}/seq/{_id}.npy", seq)
        tab = (np.array([s_tab, s_img*0.3+s_seq*0.3+np.random.randn()*0.05] + list(np.random.rand(14)))).astype("float32")
        np.save(f"{ROOT}/{split}/tab/{_id}.npy", tab)

        rows.append({"id":_id,"label":y,"region":reg,"climate_zone":cz,"county_fips":fips})
    pd.DataFrame(rows).to_csv(f"{ROOT}/{split}/labels.csv", index=False)
    print(f"wrote {ROOT}/{split}/labels.csv with {len(rows)} rows")

if __name__ == "__main__":
    mk_dirs()
    make_split("val", N_VAL)
    make_split("test", N_TEST)
    print("✅ synthetic dataset ready in /workspace/data")

