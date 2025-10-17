import argparse, yaml, os, json, csv
def main(cfg):
    os.makedirs("out", exist_ok=True)
    fusion = json.load(open("out/fusion_results.json"))
    with open("out/fusion_table.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["Model","AUROC","Brier","ECE","Acc","F1"])
        for name in ["best_metrics","weighted","stacking"]:
            m = fusion["weighted"] if name=="weighted" else (fusion["best_metrics"] if name=="best_metrics" else fusion["stacking"])
            w.writerow([name, m["auroc"], m["brier"], m["ece"], m["acc"], m["f1"]])
    print("Wrote out/fusion_table.csv")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)

