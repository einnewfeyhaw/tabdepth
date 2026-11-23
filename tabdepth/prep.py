"""Build the shared benchmark file once: images, GT and nested point orders, so all envs get identical inputs."""
import sys
import h5py, numpy as np, scipy.io
from PIL import Image
from huggingface_hub import hf_hub_download
from common import H, W
out = sys.argv[1]
mat = hf_hub_download("andrew-healey/nyuv2", "nyu_depth_v2_labeled.mat", revision="742f4d8a35f024263c6a9f7f316954b06ec5a817", repo_type="model")
ids = scipy.io.loadmat(sys.argv[2])["testNdxs"].ravel().astype(int) - 1
assert len(ids) == 654
f = h5py.File(mat, "r")
rgbs, gts, cr, cc = [], [], [], []
yy, xx = np.mgrid[0:H, 0:W]; co = np.stack([yy / (H - 1), xx / (W - 1)], -1).reshape(-1, 2)
for k, i in enumerate(ids):
    rgb = np.asarray(f["images"][i]).transpose(2, 1, 0); d = np.asarray(f["depths"][i]).T.astype(np.float32)
    rgb = np.asarray(Image.fromarray(rgb).resize((320, 240), Image.BILINEAR))[6:6 + H, 8:8 + W]
    d = np.asarray(Image.fromarray(d, mode="F").resize((320, 240), Image.BILINEAR))[6:6 + H, 8:8 + W]
    rgbs.append(rgb); gts.append(d)
    rng = np.random.default_rng(10_000 + int(i)); cand = np.flatnonzero(d.reshape(-1) > 0)
    cr.append(rng.permutation(cand)[:500])
    cen = co[rng.choice(cand, 3, replace=False)]
    d2 = ((co[cand, None] - cen[None]) ** 2).sum(-1).min(1); w = np.exp(-d2 / (2 * .07 ** 2)) + 1e-5
    keys = -np.log(np.clip(rng.random(len(cand)), 1e-12, 1)) / w
    cc.append(cand[np.argsort(keys)[:500]])
np.savez_compressed(out, rgb=np.stack(rgbs), gt=np.stack(gts), ctx_random=np.stack(cr).astype(np.int32), ctx_clustered=np.stack(cc).astype(np.int32), ids=ids)
print("prep done", len(ids), float(np.stack(gts).min()), float(np.stack(gts).max()))
