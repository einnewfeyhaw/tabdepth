"""Shared benchmark definition. Python 3.8 compatible: every method env imports this.

Datasets
- nyu:   NYUv2 official 654 test images; 640x480 -> 320x240 bilinear -> centre crop 304x228; dense (filled) GT;
         input points sampled from GT (random / clustered, nested), as in NLSPN-style evaluation.
- kitti: KITTI depth completion val_selection_cropped, 1000 frames, 352x1216; semi-dense GT;
         input points are subsets of the raw LiDAR (velodyne_raw), nested random subsets of 500 / 2000, or all LiDAR points.
Every method receives exactly the same (index, value) pairs and is scored on the same GT pixels.
"""
import csv, os, glob, json, hashlib
import numpy as np
from PIL import Image

# seconds = correction-only time for our residual methods (backbone and alignment excluded; those are t_backbone, t_align)
# and end-to-end network time for the specialists (their backbone runs inside the timed call).
FIELDS = ["image", "pattern", "n", "backbone", "method", "rmse", "mae", "rel", "rmse_unobs", "mae_unobs", "seconds", "peak_gb",
          "n_ctx", "n_eval", "retained_ctx", "t_backbone", "t_align", "timing"]
NYU_CONDS = [("random", 500), ("random", 200), ("random", 100), ("random", 50), ("clustered", 500), ("clustered", 100)]
KITTI_CONDS = [("lidar", 500), ("lidar", 2000), ("lidar", 0)]  # n=0 means every LiDAR point


class NYU:
    name, H, W, dmax, conds = "nyu", 228, 304, 10.0, NYU_CONDS

    def __init__(self, path):
        z = np.load(path)
        self._rgb, self._gt = z["rgb"], z["gt"]
        self._ctx = {"random": z["ctx_random"], "clustered": z["ctx_clustered"]}
        self.n_images = len(self._rgb)

    def rgb(self, i): return self._rgb[i]
    def gt(self, i): return self._gt[i].reshape(-1).astype(np.float64)
    def scene(self, i): return str(i)

    def order(self, i, pattern, seed=0):
        """Point order. seed 0 = the stored benchmark order; other seeds regenerate with the same recipe as prep.py."""
        d = self._gt[i].reshape(-1)
        if pattern.startswith("sr"):   # low-resolution depth: the value AT the centre pixel of every k x k block (point sampling, not block averaging)
            k = int(pattern[2:]); yy, xx = np.mgrid[k // 2:self.H:k, k // 2:self.W:k]; c = (yy * self.W + xx).reshape(-1); return c[d[c] > 0]
        if pattern.startswith("hole"):   # everything observed except one square hole of side W/4 (76 px; 160 px at 640 wide), position drawn per image
            # hole = W/4 (76 px), hole8 = W/8 (38 px), hole2 = W/2 (152 px)
            hs = self.W // int(pattern[4:] or 4); rng = np.random.default_rng(30_000 + int(i)); y0, x0 = int(rng.integers(0, self.H - hs + 1)), int(rng.integers(0, self.W - hs + 1))
            m = np.ones((self.H, self.W), bool); m[y0:y0 + hs, x0:x0 + hs] = False; return np.flatnonzero(m.reshape(-1) & (d > 0))
        if seed == 0: return self._ctx[pattern][i]
        cand = np.flatnonzero(d > 0)
        rng = np.random.default_rng(10_000 + int(i) + 1_000_003 * int(seed))
        if pattern == "random": return rng.permutation(cand)[:500]
        yy, xx = np.mgrid[0:self.H, 0:self.W]; co = np.stack([yy / (self.H - 1), xx / (self.W - 1)], -1).reshape(-1, 2)
        cen = co[rng.choice(cand, 3, replace=False)]
        d2 = ((co[cand, None] - cen[None]) ** 2).sum(-1).min(1); w = np.exp(-d2 / (2 * .07 ** 2)) + 1e-5
        return cand[np.argsort(-np.log(np.clip(rng.random(len(cand)), 1e-12, 1)) / w)[:500]]

    def sample(self, i, pattern, n, noise="clean", seed=0):
        """noise: clean | gS (multiplicative log-normal, sigma S/100) | oF (F percent of points replaced by uniform depth in [0.5, 8] m)."""
        c = self.order(i, pattern, seed); c = c[:n] if n > 0 else c; obs = self.gt(i)[c].copy()   # n = 0: the whole pattern (sr*, hole)
        if noise[0] in "ic" and noise[1:].isdigit():   # structured-corruption control: P percent of the points are made 1.5x deeper. iP = independent random points; cP = the same number of points in spatial clusters of about 10
            P = int(noise[1:]); rng = np.random.default_rng(900_000 + 31 * int(i) + P); k = int(round(P / 100.0 * len(c)))
            if noise[0] == "i": idx = rng.choice(len(c), k, replace=False)
            else:
                from scipy.spatial import cKDTree
                tr = cKDTree(np.column_stack([c // self.W, c % self.W])); seeds = rng.choice(len(c), max(1, k // 10), replace=False); nn = tr.query(tr.data[seeds], k=min(len(c), 40))[1]
                idx = []; seen = set()
                for col in range(nn.shape[1]):   # grow all clusters one neighbour at a time until k points are corrupted
                    for r in range(nn.shape[0]):
                        if len(idx) < k and int(nn[r, col]) not in seen: seen.add(int(nn[r, col])); idx.append(int(nn[r, col]))
                idx = np.array(idx, dtype=int)
            obs[idx] *= 1.5; return c, obs
        if noise != "clean":
            rng = np.random.default_rng(777 + 31 * int(i) + 7919 * int(noise[1:]) + (0 if noise[0] == "g" else 100_000))
            if noise[0] == "g": obs *= np.exp(rng.normal(0, int(noise[1:]) / 100.0, len(obs)))
            elif noise[0] == "o":
                k = int(round(int(noise[1:]) / 100.0 * len(obs))); idx = rng.choice(len(obs), k, replace=False); obs[idx] = rng.uniform(0.5, 8.0, k)
            else: raise ValueError(noise)
        return c, obs


class KITTI:
    name, H, W, dmax, conds = "kitti", 352, 1216, 100.0, KITTI_CONDS

    def __init__(self, root, ctx_path):
        d = os.path.join(root, "depth_selection", "val_selection_cropped")
        self.img = sorted(glob.glob(os.path.join(d, "image", "*.png")))
        assert len(self.img) == 1000, len(self.img)
        self.lid = [p.replace("/image/", "/velodyne_raw/").replace("_sync_image_", "_sync_velodyne_raw_") for p in self.img]
        self.gtp = [p.replace("/image/", "/groundtruth_depth/").replace("_sync_image_", "_sync_groundtruth_depth_") for p in self.img]
        z = np.load(ctx_path)
        self._ctx = z["ctx"]  # (1000, 2000) nested random order over valid LiDAR pixels
        self.n_images = len(self.img)

    @staticmethod
    def _depth(p):
        a = np.asarray(Image.open(p), dtype=np.float64)  # uint16 png, metres * 256, 0 = missing
        return (a / 256.0).reshape(-1)

    def rgb(self, i):
        a = np.asarray(Image.open(self.img[i]).convert("RGB"))
        assert a.shape == (self.H, self.W, 3), a.shape
        return a

    def gt(self, i): return self._depth(self.gtp[i])
    def lidar(self, i): return self._depth(self.lid[i])
    def scene(self, i): return os.path.basename(self.img[i]).split("_sync_")[0]

    def sample(self, i, pattern, n, noise="clean", seed=0):
        """noise (applied to the LiDAR values passed as input; GT untouched, positions unchanged):
        gS = every value multiplied by exp(N(0, (S/100)^2)); oF = round(F% of the points) chosen uniformly, value replaced by U[1, 80] m.
        RNG seeded per frame and corruption type, identical for every method."""
        assert pattern == "lidar" and seed == 0
        l = self.lidar(i)
        c = np.flatnonzero(l > 0) if n == 0 else self._ctx[i][:n]
        assert (l[c] > 0).all()
        obs = l[c].copy()
        if noise != "clean":
            rng = np.random.default_rng(777 + 31 * int(i) + 7919 * int(noise[1:]) + (0 if noise[0] == "g" else 100_000) + 13 * int(n))
            if noise[0] == "g": obs *= np.exp(rng.normal(0, int(noise[1:]) / 100.0, len(obs)))
            elif noise[0] == "o":
                k = int(round(int(noise[1:]) / 100.0 * len(obs))); idx = rng.choice(len(obs), k, replace=False); obs[idx] = rng.uniform(1.0, 80.0, k)
            else: raise ValueError(noise)
        return c, obs


class Uni:
    """One benchmark setting of the OMNI-DC 'uniformat' evaluation release, converted by uni_prep.py to pickle-free .npz (rgb, gt, dep).
    The released sparse depth map IS the input (same points and values as the published tables); image size can vary per image."""
    name, conds = "uni", [("uni", 0)]

    def __init__(self, path):
        self.files = sorted(glob.glob(os.path.join(path, "*.npz"))); assert self.files, path
        self.n_images = len(self.files); self.tag = os.path.basename(os.path.normpath(path)); self._k = None

    def _z(self, i):
        if self._k != i:
            z = np.load(self.files[i], allow_pickle=False); self._d = (z["rgb"], z["gt"].astype(np.float64), z["dep"].astype(np.float64)); self._k = i
        return self._d

    def rgb(self, i): return self._z(i)[0]
    def shape(self, i): return self._z(i)[1].shape
    def gt(self, i): g = self._z(i)[1].reshape(-1).copy(); g[g <= 1e-4] = 0; return g   # OMNI-DC valid mask: gt > 1e-4
    def scene(self, i): return str(i)

    def sample(self, i, pattern="uni", n=0, noise="clean", seed=0):
        """clean = the released sparse map. gS / oF = our extra corruption on top (as KITTI.sample; outliers ~ U[min, max] of the image's own points)."""
        d = self._z(i)[2].reshape(-1); c = np.flatnonzero(d > 0); obs = d[c].copy()
        if noise == "gt":   # control: same pixel locations, values replaced by ground truth (points without ground truth are dropped): isolates layout from measurement error
            g = self.gt(i); k = g[c] > 0; return c[k], g[c][k]
        if noise == "rnd":  # control: as many points as released, at uniformly random pixels that have ground truth, with ground-truth values (layout AND values replaced)
            g = self.gt(i); v = np.flatnonzero(g > 0); k = np.random.default_rng(40_000 + int(i)).choice(v, min(len(c), len(v)), replace=False); return np.sort(k), g[np.sort(k)]
        if noise == "pos":  # companion control: the released values, restricted to the same subset of points that have ground truth
            g = self.gt(i); k = g[c] > 0; return c[k], obs[k]
        if noise[0] == "k":   # occlusion test for projected LiDAR, no ground truth used: "kWY-WX-T" drops a point deeper than (1 + T/100) x the nearest point within a (2WY+1) x (2WX+1) window
            from scipy.ndimage import minimum_filter
            wy, wx, tau = map(int, noise[1:].split("-")); D = self._z(i)[2]; mn = minimum_filter(np.where(D > 0, D, np.inf), size=(2 * wy + 1, 2 * wx + 1), mode="constant", cval=np.inf)
            k = (D[D > 0] <= mn[D > 0] * (1 + tau / 100.0)); return c[k], obs[k]
        if noise != "clean":
            rng = np.random.default_rng(777 + 31 * int(i) + 7919 * int(noise[1:]) + (0 if noise[0] == "g" else 100_000))
            if noise[0] == "g": obs *= np.exp(rng.normal(0, int(noise[1:]) / 100.0, len(obs)))
            elif noise[0] == "o":
                k = int(round(int(noise[1:]) / 100.0 * len(obs))); idx = rng.choice(len(obs), k, replace=False); obs[idx] = rng.uniform(d[c].min(), d[c].max(), k)
            else: raise ValueError(noise)
        return c, obs


def get_dataset(name, data, ctx=None):
    if name == "uni": return Uni(data)
    ds = NYU(data) if name == "nyu" else KITTI(data, ctx); ds.shape = lambda i: (ds.H, ds.W); return ds


def parse_conds(s, default):
    if not s: return list(default)
    return [(c.split(":")[0], int(c.split(":")[1])) for c in s.split(",")]


def metrics(pred, gt, ctx):
    pred = np.asarray(pred, dtype=np.float64).reshape(-1); t = np.asarray(gt, dtype=np.float64).reshape(-1)
    assert pred.shape == t.shape
    v = t > 0
    assert np.isfinite(pred[v]).all()
    u = v.copy(); u[ctx] = False
    e = pred - t
    return dict(rmse=float(np.sqrt(np.mean(e[v] ** 2))), mae=float(np.mean(np.abs(e[v]))), rel=float(np.mean(np.abs(e[v]) / t[v])),
                rmse_unobs=float(np.sqrt(np.mean(e[u] ** 2))), mae_unobs=float(np.mean(np.abs(e[u]))))


class Writer:
    """Appends rows; existing rows make a resumed spot run skip finished work."""
    def __init__(self, path):
        self.path = path; self.done = set()
        new = not os.path.exists(path); fields = FIELDS
        if not new:
            with open(path) as f:
                rd = csv.DictReader(f); fields = rd.fieldnames or FIELDS  # keep a resumed file's own header
                for r in rd:
                    self.done.add((int(r["image"]), r["pattern"], int(r["n"]), r["backbone"], r["method"]))
        self.f = open(path, "a", newline=""); self.w = csv.DictWriter(self.f, fields, extrasaction="ignore", restval="")
        if new: self.w.writeheader()

    def has(self, image, pattern, n, backbone, method): return (image, pattern, n, backbone, method) in self.done

    def add(self, image, pattern, n, backbone, method, pred, gt, ctx, seconds, peak_gb=float("nan"), **extra):
        extra.setdefault("n_ctx", len(ctx)); extra.setdefault("n_eval", int((np.asarray(gt) > 0).sum()))
        self.w.writerow(dict(image=image, pattern=pattern, n=n, backbone=backbone, method=method, seconds=seconds, peak_gb=peak_gb,
                             **extra, **metrics(pred, gt, ctx)))
        self.f.flush(); self.done.add((image, pattern, n, backbone, method))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def provenance(out_csv, info):
    """Record exact checkpoints and preprocessing next to the results file."""
    with open(out_csv.replace(".csv", ".provenance.json"), "w") as f: json.dump(info, f, indent=1, default=str)


def shard(n_images, i, k, limit=0):
    idx = list(range(n_images))[i::k]
    return idx[:limit] if limit else idx
