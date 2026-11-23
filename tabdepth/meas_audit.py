"""Audit of the released input points against ground truth, per benchmark setting (no model involved).  meas_audit.py UNI_ROOT OUT_CSV setting [setting ...]
Per image: number of points, share that fall on a pixel with ground truth, and for those the RMSE, median and 90th percentile of |point - gt| / gt, share with > 10% error;
plus coverage: median and 90th percentile over GT pixels of the distance (px) to the nearest point."""
import sys, glob, os, csv
import numpy as np
from scipy.spatial import cKDTree
root, out = sys.argv[1], sys.argv[2]; w = csv.writer(open(out, "w", newline="")); w.writerow(["setting", "image", "n_pts", "frac_with_gt", "rmse_m", "med_rel", "p90_rel", "frac_gt10pct", "dist_med_px", "dist_p90_px", "gt_med_m"])
for st in sys.argv[3:]:
    for k, f in enumerate(sorted(glob.glob(os.path.join(root, st, "*.npz")))):
        z = np.load(f, allow_pickle=False); g, d = z["gt"].astype(np.float64), z["dep"].astype(np.float64); H, W = g.shape; v = g > 1e-4; s = d > 0; b = s & v
        if s.sum() == 0: continue
        rel = np.abs(d[b] - g[b]) / g[b] if b.any() else np.array([np.nan])
        ys, xs = np.nonzero(v); sub = slice(None, None, max(1, len(ys) // 20000)); dist = cKDTree(np.column_stack(np.nonzero(s))).query(np.column_stack([ys[sub], xs[sub]]))[0]
        w.writerow([st, k, int(s.sum()), float(b.sum() / s.sum()), float(np.sqrt(np.mean((d[b] - g[b]) ** 2))) if b.any() else "", float(np.median(rel)), float(np.quantile(rel, .9)), float((rel > .1).mean()),
                    float(np.median(dist)), float(np.quantile(dist, .9)), float(np.median(g[v]))])
    print("audited", st, flush=True)
