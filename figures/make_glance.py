"""Benchmark at a glance: our RMSE divided by the published OMNI-DC RMSE for every benchmark setting (values < 1 = ours better).   make_glance.py OUT_DIR
Numbers are copied from the results tables (bench/results/bench_tables.txt, published_baselines.md); update here when a table changes."""
import sys, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix", "font.size": 7.5, "axes.linewidth": .6, "pdf.fonttype": 42})
G = [  # (group, [(label, ours prior-assisted, ours direct or None, OMNI-DC)])
 ("NYUv2\n(points)", [("500", .108, .117, .111), ("200", .142, .167, .147), ("100", .169, .223, .180), ("50", .202, .308, .225), ("5", .357, .956, .536)]),
 ("VOID", [("1500", .525, .541, .555), ("500", .512, None, .551), ("150", .587, None, .650)]),
 ("Synthetic patterns,\nindoor (4 datasets)", [("0.7%", .076, .093, .084), ("0.1%", .108, .181, .151), ("0.03%", .142, None, .233), ("noise 5%", .078, None, .090), ("noise 10%", .080, None, .097), ("ORB", .139, None, .240), ("SIFT", .120, None, .203), ("L-16", .089, None, .107), ("L-8", .110, None, .163)]),
 ("Synthetic patterns,\noutdoor (2 datasets)", [("0.7%", 1.186, 1.213, 1.275), ("0.1%", 1.784, 1.963, 1.889), ("0.03%", 2.227, None, 2.477), ("noise 5%", 1.220, None, 1.323), ("noise 10%", 1.226, None, 1.390), ("ORB", 1.369, None, 1.646), ("SIFT", 1.368, None, 1.43), ("L-64", 1.078, None, 1.271), ("L-16", 1.421, None, 1.513), ("L-8", 1.820, None, 2.096)]),
 ("Real SfM\n(ETH3D)", [("indoor", .664, None, .605), ("outdoor", 1.813, None, 1.069)]),
 ("Real LiDAR\n(KITTI)", [("64", 1.498, None, 1.191, 1.321), ("16", 2.363, None, 1.682, 2.186), ("8", 3.142, None, 2.058, 2.986)]),
]
fig, ax = plt.subplots(figsize=(6.9, 1.75)); x = 0; ticks, labs = [], []
for gi, (g, rows) in enumerate(G):
    x0 = x
    for lab, p, d, o, *f in rows:
        if f: ax.plot([x - .31, x + .31], [f[0] / o - 1] * 2, color="k", lw=1.3, zorder=5, solid_capstyle="butt")
        ax.bar(x, p / o - 1, width=.62, color="#0072B2" if p <= o else "#B2182B", zorder=3)
        if d is not None: ax.plot(x, min(d / o - 1, .86), marker="o", ms=3.2, mfc="w", mec="#D55E00", mew=.9, zorder=4, ls="none")
        ticks.append(x); labs.append(lab); x += 1
    ax.text((x0 + x - 1) / 2, .93, g, ha="center", va="top", fontsize=7, linespacing=.95)
    if gi < len(G) - 1: ax.axvline(x - .1, color="#BBBBBB", lw=.5, zorder=1)
    x += .8
ax.axhline(0, color="k", lw=.6, zorder=2); ax.set_ylim(-.66, .95); ax.set_xlim(-.7, x - 1); ax.set_xticks(ticks); ax.set_xticklabels(labs, rotation=55, ha="right", rotation_mode="anchor", fontsize=6.3)
ax.set_yticks([-.4, -.2, 0, .2, .4, .6]); ax.set_yticklabels(["$-40\\%$", "$-20\\%$", "0", "$+20\\%$", "$+40\\%$", "$+60\\%$"]); ax.set_ylabel("RMSE relative to\nOMNI-DC (published)", fontsize=7)
ax.spines[["top", "right"]].set_visible(False); ax.tick_params(length=2, width=.5, pad=1.5); ax.grid(axis="y", lw=.3, color="#E3E3E3", zorder=0)
ax.plot([], [], "s", color="#0072B2", ms=4, label="Ours with a frozen depth prior: lower error"); ax.plot([], [], "s", color="#B2182B", ms=4, label="higher error"); ax.plot([], [], "-", color="k", lw=1.3, label="with LiDAR pre-filter"); ax.plot([], [], "o", mfc="w", mec="#D55E00", ms=3.5, label="Ours, no depth-trained network")
h, l = ax.get_legend_handles_labels(); o_ = [0, 1, 3, 2]; h = [h[i] for i in o_]; l = [l[i] for i in o_]
ax.legend(h, l, loc="lower left", ncol=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0, -.04), handletextpad=.2, columnspacing=1.0)
plt.subplots_adjust(left=.095, right=.998, top=.985, bottom=.24); out = sys.argv[1]; fig.savefig(os.path.join(out, "glance.pdf")); os.system(f"sips -s format png {out}/glance.pdf --out {out}/glance.png >/dev/null 2>&1")
