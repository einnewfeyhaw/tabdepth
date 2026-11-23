"""Benchmark table from run_ours.py outputs on the uniformat sets.  analyze_uni.py DIR -> per-setting mean RMSE / MAE / REL per method, images done / expected."""
import sys, glob, os
import numpy as np, pandas as pd
pd.set_option("display.width", 300, "display.max_columns", 60, "display.max_rows", 400)
EXP = {"ARKitScenes": 800, "iBims": 100, "ETH3D_Indoor": 193, "ETH3D_Outdoor": 197, "DIODE_Indoor": 325, "DIODE_Outdoor": 446, "KITTIDC_test": 749, "KITTIDC_val": 251,
       "NYU": 654, "VOID": 800, "ETH3D_SfM_Indoor_test": 193, "ETH3D_SfM_Outdoor_test": 197, "ETH3D_SfM_Indoor_val": 26}
df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{sys.argv[1]}/ours_*.csv") if os.path.getsize(f) > 200]).drop_duplicates(["image", "pattern", "method"])
keep = ["kumo/med", "frozen/med", "icl/med", "pfn/med", "basic/med", "mp/med", "knn/mean", "gbt/mean", "zero/mean"]; df = df[df.method.isin(keep)]
n = df.groupby(["pattern", "method"]).image.nunique().groupby("pattern").max(); exp = n.index.map(lambda p: next((v for k, v in sorted(EXP.items(), key=lambda kv: -len(kv[0])) if p.startswith(k)), -1))
for m in ("rmse", "mae", "rel"):
    t = df.pivot_table(index="pattern", columns="method", values=m).reindex(columns=[k for k in keep if k in df.method.unique()]).round(4)
    t["max_done"] = [f"{a}/{b}" for a, b in zip(n.reindex(t.index), exp)]; cnt = df.pivot_table(index="pattern", columns="method", values="image", aggfunc="nunique"); t = t.where(cnt.reindex(columns=t.columns[:-1]).ge(pd.Series(exp, index=n.index), axis=0).reindex(columns=t.columns, fill_value=True)); print(f"\n=== {m.upper()} ==="); print(t.to_string())
print("\n=== correction seconds ==="); print(df.pivot_table(index="pattern", columns="method", values="seconds").round(1).to_string())
fb = df[df.timing.astype(str).str.startswith("fallback")]
if len(fb): print("\nfallback (<3 points) images:", fb.groupby("pattern").image.nunique().to_dict())
