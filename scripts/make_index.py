import os, glob, pandas as pd, argparse

def build_index(root, split, zfill_width=4):
    base = os.path.join(root, split)
    imgs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(f"{base}/images/*")}
    seqs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(f"{base}/seq/*")}
    tabs = {os.path.splitext(os.path.basename(p))[0]: p
            for p in glob.glob(f"{base}/tab/*")}

    # preserve leading zeros
    lab  = pd.read_csv(os.path.join(base, "labels.csv"), dtype={"id": str})
    rows, missing = [], []

    for _, r in lab.iterrows():
        _id = str(r["id"]).zfill(zfill_width)
        img, seq, tab = imgs.get(_id), seqs.get(_id), tabs.get(_id)
        if not (img and seq and tab):
            missing.append((_id, bool(img), bool(seq), bool(tab)))
            continue
        rows.append({
            "id": _id, "image_path": img, "seq_path": seq, "feat_path": tab,
            "label": int(r["label"]),
            "region": r.get("region","NA"),
            "climate_zone": r.get("climate_zone","NA"),
            "county_fips": r.get("county_fips","NA"),
        })

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(root, f"{split}_index.csv"), index=False)
    print(f"wrote {split}_index.csv rows={len(out)}")
    if missing:
        print(f"⚠️ Skipped {len(missing)} IDs without full triplet. First 10:")
        for m in missing[:10]:
            _id, has_img, has_seq, has_tab = m
            print(f"  id={_id}  image={has_img}  seq={has_seq}  tab={has_tab}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--zfill", type=int, default=4,
                    help="zero-pad width for IDs (default 4)")
    args = ap.parse_args()
    build_index(args.root, "val",  args.zfill)
    build_index(args.root, "test", args.zfill)
