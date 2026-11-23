"""Paper figures from saved predictions and diagnostics.   make_figs.py NYU_BENCH_NPZ PRED_DIR OUT_DIR
PRED_DIR holds diag_nyu_diag-*.predNNNNN.npz and diag_nyu_diag-*.diag.csv (from run_ours.py --diag 1 --save N).
Depth uses the Spectral colormap (near = red, far = blue), one range per row taken from the ground truth; errors use magma."""
import sys, glob, re, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": .6, "pdf.fonttype": 42})
bench, pd_dir, out = sys.argv[1:4]; H, W = 228, 304; B = np.load(bench); RGB = B["rgb"]
P = {int(re.search(r"pred(\d+)", f).group(1)): f for f in glob.glob(os.path.join(pd_dir, "diag_nyu_diag-*.pred*.npz"))}
C = {"prior": "#0072B2", "direct": "#D55E00", "knn": "#009E73", "base": "#7F7F7F"}   # colour-blind-safe (Okabe-Ito)

def rmse(a, g): v = g > 0; return float(np.sqrt(np.mean((a[v] - g[v]) ** 2)))
def show(ax, img, **kw): ax.imshow(img, interpolation="nearest", **kw); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]
def points(ax, i, c, obs, vmin, vmax, s=2.2):
    show(ax, RGB[i]); ax.scatter(c % W, c // W, c=obs, cmap="Spectral", vmin=vmin, vmax=vmax, s=s, linewidths=.15, edgecolors="k")
def depth(ax, d, g, vmin, vmax, tag=True):
    show(ax, d.reshape(H, W), cmap="Spectral", vmin=vmin, vmax=vmax)
    if tag: ax.text(.985, .03, f"{rmse(d, g):.3f} m", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color="k", bbox=dict(boxstyle="round,pad=.15", fc="w", ec="none", alpha=.8))

def teaser(ids, name):
    cols = ["Image and 500 measured depths", "Ours, no depth-trained network", "Ours, with a frozen depth prior", "Ground truth"]
    fig, ax = plt.subplots(len(ids), 4, figsize=(6.9, 1.18 * len(ids) + .18))
    for r, i in enumerate(ids):
        z = np.load(P[i]); g = z["gt"].astype(np.float32); lo, hi = np.percentile(g[g > 0], [1, 99])
        points(ax[r, 0], i, z["ctx|random500"], z["obs|random500"], lo, hi); depth(ax[r, 1], z["nd_dino_rgb|random500"].astype(np.float32), g, lo, hi)
        depth(ax[r, 2], z["kumo|random500"].astype(np.float32), g, lo, hi); depth(ax[r, 3], g, g, lo, hi, tag=False)
    for a, t in zip(ax[0], cols): a.set_title(t, fontsize=8, pad=3)
    plt.subplots_adjust(left=.004, right=.996, top=1 - .16 / (1.18 * len(ids) + .18), bottom=.004, wspace=.02, hspace=.03); fig.savefig(os.path.join(out, name)); plt.close(fig)

def qualitative(rows, name):
    """rows: (image id, setting key, row label). Columns: input, kNN on the same columns, direct, prior-assisted, ground truth, error of prior-assisted."""
    cols = ["Input", r"$k$-NN, same columns", "Ours, direct", "Ours, prior-assisted", "Ground truth", "Abs. error (prior-assisted)"]
    fig, ax = plt.subplots(len(rows), 6, figsize=(6.9, .89 * len(rows) + .2))
    for r, (i, key, lab) in enumerate(rows):
        z = np.load(P[i]); g = z["gt"].astype(np.float32); lo, hi = np.percentile(g[g > 0], [1, 99]); sfx = "@o20" if "@o20" in key else ""; st = key.replace("@o20", "")
        k = lambda m: z[f"{m}{sfx}|{st}"].astype(np.float32)
        points(ax[r, 0], i, z[f"ctx|{key}"], z[f"obs|{key}"], lo, hi, s=1.6 if "500" in key else 5); ax[r, 0].set_ylabel(lab, fontsize=7, labelpad=2)
        depth(ax[r, 1], k("knn"), g, lo, hi); depth(ax[r, 2], k("nd_dino_rgb"), g, lo, hi); depth(ax[r, 3], k("kumo"), g, lo, hi); depth(ax[r, 4], g, g, lo, hi, tag=False)
        show(ax[r, 5], np.abs(k("kumo") - g).reshape(H, W), cmap="magma", vmin=0, vmax=.5)
    for a, t in zip(ax[0], cols): a.set_title(t, fontsize=7, pad=2.5)
    plt.subplots_adjust(left=.03, right=.998, top=1 - .15 / (.89 * len(rows) + .2), bottom=.004, wspace=.02, hspace=.03); fig.savefig(os.path.join(out, name)); plt.close(fig)

def distance(name):
    D = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(pd_dir, "diag_nyu_diag-*.diag.csv"))]).drop_duplicates(["image", "pattern", "n", "method", "kind", "bin_lo"])
    D = D[D.kind == "dist"]; D["set"] = D.pattern + D.n.astype(str); g = D.groupby(["set", "method", "bin_lo"]).agg(cnt=("count", "sum"), sse=("sse", "sum")).reset_index(); g["rmse"] = np.sqrt(g.sse / g.cnt)
    mid = {-1.0: .35, .5: 1, 1.5: 2.1, 3.0: 4.2, 6.0: 8.5, 12.0: 17, 24.0: 34, 48.0: 68, 96.0: 130}
    fig, ax = plt.subplots(1, 3, figsize=(6.9, 1.75))
    for a, (s, t) in zip(ax, [("random500", "500 random points"), ("random50", "50 random points"), ("clustered500", "500 clustered points")]):
        tot = g[(g.set == s) & (g.method == "kumo")].cnt.sum()
        for m, lab, c, ls, mk in (("zero", "Aligned depth prior only", C["base"], ":", "s"), ("knn", r"$k$-NN, same columns as prior-assisted", C["knn"], "-.", "^"),
                                  ("nd_dino_rgb", "Ours, direct (no depth-trained network)", C["direct"], "--", "o"), ("kumo", "Ours, prior-assisted", C["prior"], "-", "D")):
            x = g[(g.set == s) & (g.method == m) & (g.cnt > .002 * tot)].sort_values("bin_lo"); a.plot([mid[b] for b in x.bin_lo], x.rmse, ls, color=c, marker=mk, ms=2.8, lw=1.1, label=lab)
        a.set_xscale("log"); a.set_title(t, fontsize=8, pad=3); a.set_xlabel("Distance to nearest measurement (px)"); a.set_xticks([1, 4, 16, 64]); a.set_xticklabels(["1", "4", "16", "64"]); a.minorticks_off()
        a.spines[["top", "right"]].set_visible(False); a.set_ylim(0, {"random500": .45, "random50": .7, "clustered500": 1.45}[s]); a.tick_params(length=2, width=.5); a.grid(axis="y", lw=.3, color="#DDDDDD"); a.set_axisbelow(True)
    ax[0].set_ylabel("RMSE (m)"); h, l = ax[0].get_legend_handles_labels(); fig.legend(h[::-1], l[::-1], loc="upper center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(.5, 1.03), columnspacing=1.2, handlelength=2.2)
    plt.subplots_adjust(left=.065, right=.995, top=.80, bottom=.2, wspace=.2); fig.savefig(os.path.join(out, name)); plt.close(fig)

if __name__ == "__main__":
    teaser([168, 231], "teaser.pdf")
    qualitative([(357, "random500", "500 random"), (651, "clustered500", "500 clustered"), (63, "random500@o20", "500, 20% outliers")], "qualitative.pdf")
    distance("distance.pdf")
    for f in ("teaser", "qualitative", "distance"):   # previews
        os.system(f"sips -s format png {os.path.join(out, f)}.pdf --out {os.path.join(out, f)}.png >/dev/null 2>&1")
