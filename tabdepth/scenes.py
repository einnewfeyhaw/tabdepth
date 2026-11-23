"""Scene name for each official test image (for scene-level bootstrap)."""
import sys, csv, h5py, numpy as np, scipy.io
from huggingface_hub import hf_hub_download
mat = hf_hub_download("andrew-healey/nyuv2", "nyu_depth_v2_labeled.mat", revision="742f4d8a35f024263c6a9f7f316954b06ec5a817")
ids = scipy.io.loadmat(sys.argv[2])["testNdxs"].ravel().astype(int) - 1
f = h5py.File(mat, "r"); refs = np.asarray(f["scenes"]).reshape(-1)
with open(sys.argv[1], "w", newline="") as o:
    w = csv.writer(o); w.writerow(["image", "mat_index", "scene"])
    for k, i in enumerate(ids): w.writerow([k, int(i), "".join(chr(int(c)) for c in np.asarray(f[refs[i]]).reshape(-1))])
print("scenes done")
