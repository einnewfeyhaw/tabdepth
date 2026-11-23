"""Convert the OMNI-DC 'uniformat' evaluation release (pickled dict .npy per image: rgb, dep, gt, K) into pickle-free .npz files
and write a manifest. Run once on an isolated machine with python -I; every later job reads only the .npz files (allow_pickle off).
  uni_prep.py SRC_ROOT DST_ROOT MANIFEST_CSV
One sub-folder per benchmark setting (e.g. iBims_test_2150, KITTIDC_test_LiDAR_64); file order = sorted names = OMNI-DC's loader order.
"""
import sys, os, csv
import numpy as np
src, dst, man = sys.argv[1:4]; skip = set(open(sys.argv[4]).read().split()) if len(sys.argv) > 4 else set()
rows = []; bad = []
for d in sorted(os.listdir(src)):
    p = os.path.join(src, d)
    if not os.path.isdir(p) or d in skip: continue
    fs = sorted(f for f in os.listdir(p) if f.endswith(".npy"))
    if not fs: continue
    st = []
    try:
        for f in fs: np.load(os.path.join(p, f), allow_pickle=True, mmap_mode=None)   # any unreadable file -> the whole setting is reported, not converted
    except Exception as e:
        bad.append(d); print("BAD SETTING", d, f, repr(e)[:200], flush=True); continue
    os.makedirs(os.path.join(dst, d), exist_ok=True)
    for f in fs:
        z = dict(np.load(os.path.join(p, f), allow_pickle=True).item())
        rgb = np.asarray(z["rgb"], dtype=np.uint8); gt = np.asarray(z["gt"], dtype=np.float32); dep = np.asarray(z["dep"], dtype=np.float32)
        assert rgb.ndim == 3 and rgb.shape[2] == 3 and gt.shape == rgb.shape[:2] == dep.shape, (d, f, rgb.shape, gt.shape, dep.shape)
        np.savez_compressed(os.path.join(dst, d, f[:-4] + ".npz"), rgb=rgb, gt=gt, dep=dep, K=np.asarray(z.get("K", np.zeros(4)), dtype=np.float32).reshape(-1))
        v = gt > 1e-4; s = dep > 0
        st.append((gt.shape[0], gt.shape[1], int(s.sum()), float(v.mean()), float(gt[v].min()) if v.any() else 0, float(gt[v].max()) if v.any() else 0,
                   float(np.median(np.abs(dep[s & v] - gt[s & v]) / gt[s & v])) if (s & v).any() else -1, float((s & ~v).mean())))
    a = np.array(st)
    rows.append(dict(setting=d, n=len(fs), H_min=int(a[:, 0].min()), H_max=int(a[:, 0].max()), W_min=int(a[:, 1].min()), W_max=int(a[:, 1].max()),
                     pts_med=int(np.median(a[:, 2])), pts_min=int(a[:, 2].min()), pts_max=int(a[:, 2].max()), gt_valid_frac=round(a[:, 3].mean(), 3),
                     gt_min=round(a[:, 4].min(), 3), gt_max=round(a[:, 5].max(), 2), sparse_vs_gt_medrel=round(float(np.median(a[:, 6])), 4)))
    print(rows[-1], flush=True)
with open(man, "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
print("UNI PREP DONE", len(rows), "settings; bad:", bad, flush=True); sys.exit(1 if bad else 0)
