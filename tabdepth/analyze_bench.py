"""Paper tables from run_ours.py outputs on the OMNI-DC uniformat sets.   analyze_bench.py DIR
Routing rule (fixed): per image, Kumo if the image has <= 2000 input points, TabICLv2 otherwise. 'frozen' rows stand in for 'kumo' when N <= 2000
(the two are identical there; they differ only in how > 2000 points are split).
Prints: real-pattern sets, then the synthetic patterns averaged as in the OMNI-DC appendix (indoor = ARKitScenes, iBims, ETH3D-in, DIODE-in; outdoor = ETH3D-out, DIODE-out).
A cell is printed only when every image of every contributing set is done; otherwise it shows the number of images still missing."""
import sys, glob, os, re
import numpy as np, pandas as pd
pd.set_option("display.width", 320, "display.max_columns", 80, "display.max_rows", 400)
EXP = {"ARKitScenes": 800, "iBims": 100, "ETH3D_Indoor": 193, "ETH3D_Outdoor": 197, "DIODE_Indoor": 325, "DIODE_Outdoor": 446, "KITTIDC_test": 749, "KITTIDC_val": 251,
       "NYU": 654, "VOID": 800, "ETH3D_SfM_Indoor_test": 193, "ETH3D_SfM_Outdoor_test": 197}
def expn(p): return next((v for k, v in sorted(EXP.items(), key=lambda kv: -len(kv[0])) if p.startswith(k)), -1)
df = pd.concat([pd.read_csv(f) for f in glob.glob(f"{sys.argv[1]}/ours_*.csv") if os.path.getsize(f) > 200]).drop_duplicates(["image", "pattern", "method"])
df["N"] = df.timing.astype(str).str.extract(r"pts=(\d+)")[0].astype(float); df["N"] = df.N.fillna(df.n_ctx)
P = df.pivot_table(index=["pattern", "image"], columns="method", values=["rmse", "mae", "rel"])
Ng = df.groupby(["pattern", "image"]).N.max()
CUT = 10000   # routing threshold fixed on validation splits (KITTI val 8/16/32/64 lines, ETH3D SfM val): Kumo with <= 4 subsets up to 10k points, TabICLv2 above
def col(met, m): return P[(met, m)] if (met, m) in P.columns else pd.Series(np.nan, index=P.index)
def routed(met, small, big):
    s = col(met, small[0])
    for alt in small[1:]: s = s.fillna(col(met, alt).where(Ng.reindex(P.index) <= 2000))   # old frozen rows are valid only up to 2000 points
    return s.where(Ng.reindex(P.index) <= CUT, col(met, big))
ROWS = {"Ours (prior-assisted)": lambda met: routed(met, ["kumo/med", "frozen/med"], "icl/med"),
        "Ours (direct, no depth net)": lambda met: routed(met, ["nd_dino_rgb/med"], "nd_dino_rgb_icl/med"),
        "direct: position+RGB": lambda met: col(met, "nd_rgb/med"), "direct: position only": lambda met: col(met, "nd_xy/med"),
        "Kumo only": lambda met: col(met, "kumo/med").fillna(col(met, "frozen/med").where(Ng.reindex(P.index) <= 2000)), "TabICLv2 only": lambda met: col(met, "icl/med"),
        "TabPFN v2": lambda met: col(met, "pfn/med"), "no extra columns": lambda met: col(met, "basic/med"), "+ metric-depth columns": lambda met: col(met, "mp/med"),
        "kNN, same inputs": lambda met: col(met, "knn/mean"), "trees, same inputs": lambda met: col(met, "gbt/mean"),
        "direct kNN": lambda met: col(met, "nd_dr_knn/mean"), "direct trees": lambda met: col(met, "nd_dr_gbt/mean"), "aligned DA-V2 only": lambda met: col(met, "zero/mean")}
def cell(series, pat):
    x = series.xs(pat, level="pattern") if pat in series.index.get_level_values("pattern") else pd.Series(dtype=float); x = x.dropna(); miss = expn(pat) - len(x)
    return (x.mean(), 0) if miss <= 0 else (np.nan, miss)
pats = sorted(P.index.get_level_values("pattern").unique())
for met in ("rmse", "mae", "rel"):
    S_ = {r: f(met) for r, f in ROWS.items()}
    real = [p for p in pats if p.startswith(("NYU", "VOID", "KITTIDC", "ETH3D_SfM"))]
    t = pd.DataFrame({p: {r: (lambda v: f"{v[0]:.4f}" if v[1] == 0 else f"-{v[1]}")(cell(S_[r], p)) for r in ROWS} for p in real})
    print(f"\n=== {met.upper()}: real patterns (a value like -37 means 37 images still missing) ==="); print(t.T.to_string())
    for name, splits in (("INDOOR", ["ARKitScenes", "iBims", "ETH3D_Indoor", "DIODE_Indoor"]), ("OUTDOOR", ["ETH3D_Outdoor", "DIODE_Outdoor"])):
        out = {}
        for pt in ("2150", "300", "100", "noise0.05", "noise0.1", "orb", "sift", "LiDAR_64", "LiDAR_16", "LiDAR_8"):
            out[pt] = {}
            for r in ROWS:
                cs = [cell(S_[r], f"{sp}_test_{pt}") for sp in splits]; miss = sum(c[1] for c in cs)
                out[pt][r] = f"{np.mean([c[0] for c in cs]):.4f}" if miss == 0 else f"-{miss}"
        print(f"\n=== {met.upper()}: synthetic patterns, {name} average over {splits} ==="); print(pd.DataFrame(out).to_string())
print("\nPublished OMNI-DC v1.0 (appendix), RMSE: indoor .084 .151 .233 .090 .097 .240 .203 .611 .107 .163 | outdoor 1.275 1.889 2.477 1.323 1.390 1.646 1.429 1.271 1.513 2.096")
print("columns: 0.7% 0.1% 0.03% noise5 noise10 ORB SIFT LiDAR64 LiDAR16 LiDAR8")
