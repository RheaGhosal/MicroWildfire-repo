import pandas as pd, sys
df = pd.read_csv("data/train.csv")

# 1) Verify no group_id leaks across splits
leaks = df.pivot_table(index='group_id', columns='split', values='path', aggfunc='count').fillna(0)
leaky = leaks[(leaks>0).sum(axis=1) > 1]
if len(leaky):
    print("Group leakage across splits:\n", leaky.head())
    sys.exit(1)
print("No group leakage across splits.")

# 2) Check class balance per split
for s in ["train","val","test"]:
    sub = df[df.split==s]
    pos = sub.label.sum(); neg = (1-sub.label).sum()
    print(f"{s}: n={len(sub)} | pos={pos} neg={neg} pos_rate={pos/len(sub):.4f}")

# 3) (Optional) Temporal sanity: ensure no test timestamps inside train window if that’s your rule
if 'timestamp' in df.columns:
    tr_max = pd.to_datetime(df.loc[df.split=='train','timestamp']).max()
    te_min = pd.to_datetime(df.loc[df.split=='test','timestamp']).min()
    print(f"train max ts: {tr_max} | test min ts: {te_min}")

