"""Method figure built from a real example.   make_method.py NYU_BENCH_NPZ FEAT_DIR OUT_DIR [image_id]
FEAT_DIR holds feat_nyu_feat-0.predNNNNN.npz (run_ours.py --save, with feature maps)."""
import sys, os, glob
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix", "font.size": 7.5, "pdf.fonttype": 42})
bench, fd, out = sys.argv[1:4]; i = int(sys.argv[4]) if len(sys.argv) > 4 else 168; H, W = 228, 304
rgb = np.load(bench)["rgb"][i]; z = np.load(glob.glob(os.path.join(fd, f"*pred{i:05d}.npz"))[0])
gt = z["gt"].reshape(H, W); lo, hi = np.percentile(gt, [1, 99]); c = z["ctx|random500"]; obs = z["obs|random500"]
norm = lambda a: (a - np.percentile(a, 1)) / (np.percentile(a, 99) - np.percentile(a, 1) + 1e-9)
dino = np.clip(norm(z["feat|dino16"].astype(np.float32)[:, :3].reshape(H, W, 3)), 0, 1); rel = z["feat|rel"].reshape(H, W)
yy, xx = np.mgrid[0:H, 0:W]
FW, FH = 6.9, 2.05; fig = plt.figure(figsize=(FW, FH)); A = W / H
def im(x, y, w, img, title=None, dashed=False, **kw):
    h = w / A * FW / FH; ax = fig.add_axes([x, y, w, h]); ax.imshow(img, interpolation="nearest", **kw); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linewidth(.5); s.set_linestyle("--" if dashed else "-"); s.set_edgecolor("#555555")
    if title: ax.set_title(title, fontsize=6.5, pad=1.5)
    return ax, h
ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off")
def arrow(x0, y0, x1, y1): ov.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=.8, color="#333333", shrinkA=0, shrinkB=0))
# 1. input
ax, h = im(.005, .30, .165, rgb, "Image and $N$ measured depths"); ax.scatter(c % W, c // W, c=obs, cmap="Spectral", vmin=lo, vmax=hi, s=1.6, linewidths=.12, edgecolors="k")
# 2. columns (frozen encoders)
cw = .078; x0 = .207; g = .009; ys = [.585, .145]
im(x0, ys[0], cw, yy, "position $u$", cmap="gray"); im(x0 + cw + g, ys[0], cw, xx, "position $v$", cmap="gray"); im(x0 + 2 * (cw + g), ys[0], cw, rgb, "colour (3)")
im(x0 + .12 * (cw + g), ys[1], cw, dino, "image features (16)"); im(x0 + 1.88 * (cw + g), ys[1], cw, rel, "depth prior (optional)", dashed=True, cmap="Spectral_r")
ov.text(x0 + 1.5 * cw + g, .975, "Frozen encoders fill the columns", ha="center", va="top", fontsize=7)
arrow(.173, .52, .202, .52)
# 3. the table
tx, ty, tw, th = .515, .12, .215, .74; ncol = 6; nrow = 9; cwid = tw / ncol; rh = th / nrow; rng = np.random.default_rng(3)
ov.text(tx + tw / 2, .975, "One row per pixel", ha="center", va="top", fontsize=7)
heads = ["$u$", "$v$", "col.", "feat.", "prior", "depth"]
for k, hd in enumerate(heads): ov.text(tx + (k + .5) * cwid, ty + th - rh * .5, hd, ha="center", va="center", fontsize=6.2, fontweight="bold" if k == 5 else "normal")
rows = list(rng.choice(len(c), 3, replace=False)); cmap = plt.get_cmap("Spectral")
for r in range(1, nrow):
    ctx = r <= 3; y = ty + th - (r + 1) * rh
    if r in (4, 8):
        ov.text(tx + tw / 2, y + rh / 2, r"$\vdots$" + ("   measured pixels: context" if r == 4 else "   all other pixels: queries"), ha="center", va="center", fontsize=6); continue
    p = c[rows[r - 1]] if ctx else int(rng.integers(0, H * W)); py, px = p // W, p % W
    vals = [py / (H - 1), px / (W - 1)]; cols = [plt.get_cmap("gray")(vals[0]), plt.get_cmap("gray")(vals[1]), rgb[py, px] / 255., dino[py, px], plt.get_cmap("Spectral_r")(norm(rel)[py, px])]
    for k, col in enumerate(cols): ov.add_patch(Rectangle((tx + k * cwid + .003, y + .012), cwid - .006, rh - .024, fc=col, ec="#666666", lw=.3, ls="--" if k == 4 else "-"))
    if ctx: ov.add_patch(Rectangle((tx + 5 * cwid + .003, y + .012), cwid - .006, rh - .024, fc=cmap((gt[py, px] - lo) / (hi - lo)), ec="#666666", lw=.3)); ov.text(tx + 5.5 * cwid, y + rh / 2, f"{gt[py, px]:.2f}", ha="center", va="center", fontsize=5.6)
    else: ov.add_patch(Rectangle((tx + 5 * cwid + .003, y + .012), cwid - .006, rh - .024, fc="w", ec="#B2182B", lw=.6)); ov.text(tx + 5.5 * cwid, y + rh / 2, "?", ha="center", va="center", fontsize=7, color="#B2182B", fontweight="bold")
ov.add_patch(Rectangle((tx - .004, ty), tw + .008, th, fc="none", ec="#333333", lw=.6)); ov.plot([tx - .004, tx + tw + .004], [ty + th - rh, ty + th - rh], color="#333333", lw=.5)
arrow(.476, .52, .507, .52)
# 4. frozen TFM
bx, by, bw, bh = .765, .33, .085, .38; ov.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=.006,rounding_size=.012", fc="#E8F0F8", ec="#0072B2", lw=.9))
ov.text(bx + bw / 2, by + bh / 2 + .05, "Tabular\nfoundation\nmodel", ha="center", va="center", fontsize=7, fontweight="bold", linespacing=.95); ov.text(bx + bw / 2, by + .075, "frozen; trained on\nsynthetic tables", ha="center", va="center", fontsize=5.6, linespacing=.95, color="#333333")
arrow(.737, .52, .761, .52); arrow(.854, .52, .876, .52)
# 5. output
im(.88, .30, .115, z["nd_dino_rgb|random500"].astype(np.float32).reshape(H, W), "Depth for every pixel", cmap="Spectral", vmin=lo, vmax=hi)
ov.text(.9375, .17, "and a predictive\ndistribution per pixel", ha="center", va="center", fontsize=5.8, linespacing=.95, color="#333333")
fig.savefig(os.path.join(out, "method.pdf")); fig.savefig(os.path.join(out, "method.png"), dpi=220)
