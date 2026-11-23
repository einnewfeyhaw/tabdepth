"""Our method on any dataset (nyu | kitti | uni), with switchable options so one pass scores several variants on identical inputs.

Pipeline per image: frozen depth backbone (DA-V2 Large) -> relative depth map + extra columns -> alignment to the sparse points ->
tabular foundation model predicts the log-depth residual in context -> stride-2 prediction, bilinear upsampling.
Variant options (variants.json, name -> dict; defaults in DEF):
  tfm    kumo | tabicl | tabpfn | knn | gbt | zero (same inputs, classical regressor / no correction)
  prior  rel (DA-V2 relative) | mi | mo (metric DA-V2 indoor / outdoor used as the aligned map and depth column)
  ctx    local = one context per query tile (tiles [ny, nx], nloc nearest points, nglob random others); default = whole-image context / random subsets
  tfm    knnK (K neighbours), gp (Gaussian process), gbt2 (larger boosted trees): classical regressors on identical inputs
  xf     cross-fitted consistency filter: drop measurements further than xf robust standard deviations from their prediction by the other half (0 = off)
  ens    list of variant names: residual maps of the members (same aligned map) are averaged
  prior  none = no depth model at all: constant base (median of the points), columns = position + feats (dino16 = DINOv2-L token PCA, rgb)
  prior  ma = metric DA-V2, indoor or outdoor chosen per image by which agrees better with the points (after scale calibration)
  align  wlog / wdep = the inverse-depth fit weighted so that it minimises log-depth / metric-depth error to first order
  align  scale = one-parameter calibration of a metric prior (median log ratio): for very few points
  target log (default) | lin (additive residual in metres, for RMSE on far ranges)
  feats  nn = local measurement columns: inverse-distance-weighted residual of the 4 nearest points (leave-one-out at context rows),
         log distance to the nearest point, spread of the neighbours residuals; rel = DA-V2 relative depth as an extra column
  feats  subset of grad, ms, pca8, mprior (metric DA-V2 indoor+outdoor log-depth columns)
  cap,K  context cap and max number of disjoint random subsets averaged when there are more points than cap
  nest   estimators per call            align  global | local (k nearest points, distance-weighted scale+shift on the global map)
  gate   0 = off, else tau: residual multiplied by 1/(1+(d/tau)^2), d = normalised distance to the nearest point
  stride 2 (default) or 1
Rows: method = <variant>/<mean|med>[@noise]. seconds = correction only (tabular model + column construction); t_backbone, t_align separate.
Prediction is pointwise given the context, so only stride-2 cells that touch a GT-valid pixel are predicted (exact for every scored pixel).
"""
import argparse, json, time, warnings, hashlib, math, os, inspect
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from scipy.optimize import lsq_linear
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from common import get_dataset, Writer, shard
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
for a, d in [("--dataset", "nyu"), ("--data", None), ("--ctx", None), ("--out", None), ("--idfile", ""), ("--variants", "frozen"), ("--vfile", "variants.json"),
             ("--settings", "clean:uni:0")]: ap.add_argument(a, default=d)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--diag", type=int, default=0)   # 1: also write <out>.diag.csv (error by distance-to-measurement and by depth, 90% interval coverage)
ap.add_argument("--save", type=int, default=0)   # save predictions of about this many images (spread over the id list) as npz next to the csv
A = ap.parse_args()
DEV = "cuda"; ds = get_dataset(A.dataset, A.data, A.ctx); wr = Writer(A.out)
DEF = dict(tfm="kumo", feats=["grad", "ms", "pca8"], cap=2000, K=4, nest=2, align="global", k=5, gate=0, stride=2, prior="rel", target="log")
VF = json.load(open(A.vfile)); VARS = {v: {**DEF, **VF[v]} for v in A.variants.split(",")}
SET = [(s.split(":")[0], s.split(":")[1], int(s.split(":")[2])) for s in A.settings.split(",")]
LM = "depth-anything/Depth-Anything-V2-Large-hf"
NEEDR = any("ens" not in v and (v["prior"] != "none" or set(v["feats"]) & {"grad", "ms", "pca8", "rel"}) for v in VARS.values())   # direct-mode runs never load the depth network
if NEEDR: proc, dm = AutoImageProcessor.from_pretrained(LM), AutoModelForDepthEstimation.from_pretrained(LM).to(DEV).eval()
print("depth network loaded:", bool(NEEDR), flush=True)
for v in list(VARS.values()):
    for m in v.get("ens", []): VARS.setdefault("_" + m, {**DEF, **VF[m]})   # hidden members: computed, not written
TF = {v["tfm"] for v in VARS.values() if "ens" not in v}; NEEDD = any({"dino16", "dinoms"} & set(v["feats"]) for v in VARS.values()); NEEDM = any("mprior" in v["feats"] or v["prior"] in ("mi", "mo", "ma") for v in VARS.values())
if "kumo" in TF:
    import sdm; kumo = sdm.models.KumoTabular(task="regression", size="large", device=DEV).eval()
if "tabicl" in TF:
    from tabicl import TabICLRegressor; print("tabicl predict signature:", inspect.signature(TabICLRegressor.predict), "init:", inspect.signature(TabICLRegressor.__init__), flush=True)
if "tabpfn" in TF:
    from tabpfn import TabPFNRegressor; from huggingface_hub import hf_hub_download
    PCK = hf_hub_download("Prior-Labs/TabPFN-v2-reg", "tabpfn-v2-regressor.ckpt")
if NEEDD:
    from transformers import AutoModel
    DINO = AutoModel.from_pretrained("facebook/dinov2-large").to(DEV).eval()
if NEEDM:
    MP = [(AutoImageProcessor.from_pretrained(n), AutoModelForDepthEstimation.from_pretrained(n).to(DEV).eval())
          for n in ("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")]

def mde(img, H, W, seed):
    torch.cuda.synchronize(); s = time.perf_counter(); torch.manual_seed(seed)   # pca_lowrank is randomised: fixed per image
    with torch.inference_mode():
        inp = proc(images=Image.fromarray(img), return_tensors="pt").to(DEV); o = dm(**inp, output_hidden_states=True)
        rel = F.interpolate(o.predicted_depth[:, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy().reshape(-1)
        gh, gw = inp.pixel_values.shape[-2] // 14, inp.pixel_values.shape[-1] // 14; hs = o.hidden_states[-1][0]
        torch.cuda.synchronize(); t_bb = time.perf_counter() - s; s = time.perf_counter()
        pt = hs[hs.shape[0] - gh * gw:].float(); pt = pt - pt.mean(0)
        _, _, V = torch.pca_lowrank(pt, q=8, center=False); comp = (pt @ V[:, :8]).reshape(gh, gw, 8).permute(2, 0, 1)[None]
        comp = F.interpolate(comp, size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).reshape(-1, 8).cpu().numpy()
    return rel, comp.astype(np.float32), t_bb, time.perf_counter() - s
def mprior(img, H, W):
    torch.cuda.synchronize(); s = time.perf_counter(); out = []
    with torch.inference_mode():
        for p_, m_ in MP:
            d = m_(**p_(images=Image.fromarray(img), return_tensors="pt").to(DEV)).predicted_depth
            out.append(np.log(np.clip(F.interpolate(d[:, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy().reshape(-1), 1e-3, None)))
    torch.cuda.synchronize(); return np.stack(out, -1).astype(np.float32), time.perf_counter() - s
def dino_feats(img, H, W, seed, q=16):
    """DINOv2-L (self-supervised, no depth training) last-layer patch tokens -> per-image PCA(q) -> bilinear upsample."""
    torch.cuda.synchronize(); s = time.perf_counter(); torch.manual_seed(seed)
    sc = 518 / min(H, W); h, w = int(round(H * sc / 14)) * 14, int(round(W * sc / 14)) * 14
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float().to(DEV) / 255; x = F.interpolate(x, size=(h, w), mode="bicubic", align_corners=False)
    x = (x - torch.tensor([.485, .456, .406], device=DEV).view(1, 3, 1, 1)) / torch.tensor([.229, .224, .225], device=DEV).view(1, 3, 1, 1)
    with torch.inference_mode():
        tk = DINO(pixel_values=x).last_hidden_state[0, 1:].float(); tk = tk - tk.mean(0); _, _, V = torch.pca_lowrank(tk, q=q, center=False)
        cm = (tk @ V[:, :q]).reshape(h // 14, w // 14, q).permute(2, 0, 1)[None]
        comp = F.interpolate(cm, size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).reshape(-1, q)
        # feature pyramid: the first 8 components averaged over 3x3 and 7x7 token windows (about 40 and 100 pixels at the network's input size)
        pm = torch.cat([F.avg_pool2d(cm[:, :8], k, stride=1, padding=k // 2, count_include_pad=False) for k in (3, 7)], 1)
        pyr = F.interpolate(pm, size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).reshape(-1, 16)
    torch.cuda.synchronize(); return comp.cpu().numpy().astype(np.float32), time.perf_counter() - s, pyr.cpu().numpy().astype(np.float32)
def align(rel, c, obs, lo, hi, w=None):
    """Scale and shift in inverse depth. w = per-point weights: None = plain least squares in inverse depth (errors of near points dominate);
    obs = first-order log-depth error (d log d = d * d(1/d)); obs^2 = first-order metric depth error."""
    q = (rel - rel.mean()) / (rel.std() + 1e-6); D = np.column_stack([q, np.ones_like(q)]); w = np.ones(len(c)) if w is None else w / w.mean()
    co = lsq_linear(D[c] * w[:, None], w / obs, bounds=([0, -np.inf], [np.inf, np.inf])).x
    return np.clip(1 / np.maximum(D @ co, .5 * (1 / obs).min()), lo, hi)
def local_ss(base, c, obs, coord, k, lo, hi):
    """Distance-weighted scale+shift from the k nearest points. Returns (map for every pixel, leave-one-out map values at the context points):
    the residual targets use the leave-one-out values, so a point's own measurement never enters its own baseline."""
    tree = cKDTree(coord[c])
    def fit(d, nn, bq):
        w = 1 / (d + 1e-3); a = base[c][nn]; b = obs[nn]; sw = w.sum(1)
        ma, mb = (w * a).sum(1) / sw, (w * b).sum(1) / sw; va = (w * (a - ma[:, None]) ** 2).sum(1) / sw
        s = np.clip(np.where(va > 1e-6, (w * (a - ma[:, None]) * (b - mb[:, None])).sum(1) / sw / np.maximum(va, 1e-6), 1.), .2, 5)
        return np.clip(s * bq + mb - s * ma, lo, hi)
    k = min(k, len(c) - 1); d, nn = tree.query(coord, k=k); d2, nn2 = tree.query(coord[c], k=k + 1)
    return fit(d.reshape(len(coord), k), nn.reshape(len(coord), k), base), fit(d2[:, 1:], nn2[:, 1:], base[c])
def tfm_predict(kind, X, y, Q, seed, nest):
    """(mean, median) residual at rows Q, context (X, y); columns standardised on the context."""
    sc = StandardScaler().fit(X); Xs, Qs = sc.transform(X).astype(np.float32), sc.transform(Q).astype(np.float32)
    if kind == "kumo":
        T = sdm.TableTensor; names = [f"x{j}" for j in range(X.shape[1])]; tb = lambda t, nm: T(columns={"numerical": nm}, numerical=t); mean, med, qlo, qhi = [], [], [], []
        gen = torch.Generator(device=DEV).manual_seed(seed)
        try:
            with torch.inference_mode():
                kumo.fit(x=tb(torch.as_tensor(Xs, device=DEV), names), y=tb(torch.as_tensor(y, dtype=torch.float32, device=DEV).reshape(-1, 1), ["y"]), num_estimators=nest, estimator_batch_size=1, generator=gen)
                for j in range(0, len(Qs), 4096):
                    q = kumo.predict(tb(torch.as_tensor(Qs[j:j + 4096], device=DEV), names)).numerical
                    if q.ndim == 3 and q.shape[0] == 1: q = q[0]
                    assert q.shape[-1] == 999 and torch.isfinite(q).all(); qf = q.float(); mean.append(qf.mean(-1).cpu().numpy()); med.append(qf[:, 499].cpu().numpy()); qlo.append(qf[:, 49].cpu().numpy()); qhi.append(qf[:, 949].cpu().numpy())
        finally: kumo.clear()
        return np.concatenate(mean), np.concatenate(med), np.concatenate(qlo), np.concatenate(qhi)   # mean, median, 5% and 95% quantiles
    if kind == "zero": z = np.zeros(len(Q), np.float32); return z, z
    if kind.startswith("knn"):
        from sklearn.neighbors import KNeighborsRegressor
        m = KNeighborsRegressor(n_neighbors=min(int(kind[3:] or 8), len(Xs)), weights="distance").fit(Xs, y).predict(Qs); return m, m
    if kind == "gp":   # Gaussian process, anisotropic RBF + noise, hyper-parameters fitted per image by marginal likelihood
        from sklearn.gaussian_process import GaussianProcessRegressor as GPR; from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as CK
        g = GPR(CK(1.0) * RBF(np.ones(Xs.shape[1]), (1e-2, 1e3)) + WhiteKernel(1e-2, (1e-5, 1e1)), normalize_y=True, random_state=seed % 100000).fit(Xs.astype(np.float64), y)
        m = np.concatenate([g.predict(Qs[j:j + 8192].astype(np.float64)) for j in range(0, len(Qs), 8192)]); return m, m
    if kind == "gbt2":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=.05, min_samples_leaf=5, l2_regularization=1.0, random_state=seed % 100000).fit(Xs, y).predict(Qs); return m, m
    if kind == "gbt":
        from sklearn.ensemble import HistGradientBoostingRegressor
        m = HistGradientBoostingRegressor(random_state=seed % 100000).fit(Xs, y).predict(Qs); return m, m
    if kind == "tabicl":
        r = TabICLRegressor(n_estimators=nest, random_state=seed % 100000, device=DEV); r.fit(Xs, y.astype(np.float32))
        try: return np.asarray(r.predict(Qs, output_type="mean")).reshape(-1), np.asarray(r.predict(Qs, output_type="median")).reshape(-1)
        except TypeError: m = np.asarray(r.predict(Qs)).reshape(-1); return m, m
    if kind == "tabpfn":
        r = TabPFNRegressor(model_path=PCK, device=DEV, n_estimators=nest, random_state=123, ignore_pretraining_limits=True); r.fit(Xs, y)
        cat = lambda ot: np.concatenate([np.asarray(r.predict(Qs[j:j + 4096], output_type=ot)).reshape(-1) for j in range(0, len(Qs), 4096)])
        return cat("mean"), cat("median")
    raise ValueError(kind)
def need_cells(valid, H, W, H2, W2):
    """Stride-2 cells that the bilinear upsampling (align_corners=False) reads for at least one GT-valid pixel."""
    ys, xs = np.nonzero(valid.reshape(H, W)); m = np.zeros((H2, W2), bool)
    y0 = np.floor(np.clip((ys + .5) * H2 / H - .5, 0, None)).astype(int); x0 = np.floor(np.clip((xs + .5) * W2 / W - .5, 0, None)).astype(int)
    for dy in (0, 1):
        for dx in (0, 1): m[np.minimum(y0 + dy, H2 - 1), np.minimum(x0 + dx, W2 - 1)] = True
    return m.reshape(-1)

ids = [int(x) for x in open(A.idfile).read().split()] if A.idfile else list(range(ds.n_images)); t0 = time.time()
DEDGE = np.array([0.5, 1.5, 3, 6, 12, 24, 48, 96]); ZEDGE = np.array([1, 2, 3, 5, 8, 12, 20, 40, 80.])
if A.diag:
    import csv as _csv; DGF = open(A.out.replace(".csv", ".diag.csv"), "a", newline=""); DG = _csv.writer(DGF)
    if DGF.tell() == 0: DG.writerow(["image", "pattern", "n", "method", "kind", "bin_lo", "count", "sse", "sae", "extra"])
for kk, pos in enumerate(shard(len(ids), A.shard, A.nshard, A.limit)):
    i = ids[pos]
    if all(wr.has(i, pat, n, "large", f"{v}/{st}" + ("" if nz == "clean" else f"@{nz}")) for nz, pat, n in SET for v in VARS if not v.startswith("_") for st in ("mean", "med")): continue
    img = ds.rgb(i); t = ds.gt(i); H, W = ds.shape(i); valid = t > 0
    SAVE_THIS = bool(A.save) and pos % max(1, len(ids) // A.save) == 0; saved = {}
    yy, xx = np.mgrid[0:H, 0:W]; COORD = np.stack([yy / (H - 1), xx / (W - 1)], -1).reshape(-1, 2).astype(np.float32)
    rel, pca8, t_bb, t_pca = mde(img, H, W, 1234 + i) if NEEDR else (np.zeros(H * W, np.float32), np.zeros((H * W, 8), np.float32), 0., 0.)
    s = time.perf_counter(); rz = (rel - rel.mean()) / (rel.std() + 1e-6); R2 = rz.reshape(H, W); gx, gy = sobel(R2, 1) / 8, sobel(R2, 0) / 8
    F_ = {"grad": np.stack([gx, gy, np.hypot(gx, gy)], -1).reshape(-1, 3).astype(np.float32),
          "ms": np.stack([gaussian_filter(R2, 4), gaussian_filter(R2, 12)], -1).reshape(-1, 2).astype(np.float32), "pca8": pca8}
    t_f = {"pca8": t_pca}; t_f["grad"] = t_f["ms"] = (time.perf_counter() - s) / 2
    if NEEDM: F_["mprior"], t_f["mprior"] = mprior(img, H, W)
    if NEEDD: F_["dino16"], t_f["dino16"], F_["dinoms"] = dino_feats(img, H, W, 4321 + i); t_f["dinoms"] = 0.
    s_ = time.perf_counter(); im_ = img.astype(np.float32) / 255.; gray_ = im_.mean(-1)
    F_["rgbms"] = np.concatenate([np.stack([gaussian_filter(im_[..., ch], sg) for ch in range(3)], -1).reshape(-1, 3) for sg in (2, 6, 18)] +
                                 [np.sqrt(np.maximum(gaussian_filter(gray_ ** 2, 4) - gaussian_filter(gray_, 4) ** 2, 0)).reshape(-1, 1), np.hypot(sobel(gaussian_filter(gray_, 1), 0), sobel(gaussian_filter(gray_, 1), 1)).reshape(-1, 1)], 1).astype(np.float32)
    t_f["rgbms"] = time.perf_counter() - s_   # colour at three scales (9), local contrast (1), edge strength (1)
    F_["rgb"] = (img.reshape(-1, 3) / 255.).astype(np.float32); t_f["rgb"] = 0.; F_["rel"] = rz.reshape(-1, 1).astype(np.float32); t_f["rel"] = 0.; t_f["nn"] = 0.
    for nz, pat, n in SET:
        tag = "" if nz == "clean" else f"@{nz}"
        todo = [v for v in VARS if not v.startswith("_") and not all(wr.has(i, pat, n, "large", f"{v}/{st}{tag}") for st in ("mean", "med"))]
        if not todo: continue
        c, obs = ds.sample(i, pat, n, noise=nz, seed=0)
        if SAVE_THIS: saved[f"ctx|{pat}{n}{tag}"] = c.astype(np.int32); saved[f"obs|{pat}{n}{tag}"] = obs.astype(np.float32)
        dpx = None; c_given = c   # metrics (observed / unobserved split) always refer to the measurements as given
        if len(c) > 20000 and A.dataset == "nyu":   # dense input (missing-region pattern): keep the measurements on a regular 4-pixel grid; baselines receive the full map
            kp = ((c // W) % 4 == 2) & ((c % W) % 4 == 2); c, obs = c[kp], obs[kp]
        if len(c) == 0: print(f"NO INPUT POINTS id {i} {pat}{n}{tag}: nothing written", flush=True); continue
        if len(c) < 3:   # too few points for a 2-parameter fit: median scale on inverse relative depth, no correction; written for every variant
            q0 = rel - rel.min() + 1e-3 * (rel.max() - rel.min() + 1e-6); pr = 1 / q0; pr = pr * np.median(obs / pr[c]); pc = np.full(H * W, float(np.median(obs)))
            for v in todo:
                for st in ("mean", "med"): wr.add(i, pat, n, "large", f"{v}/{st}{tag}", pc if VARS[v]["prior"] == "none" else pr, t, c, 0., 0., retained_ctx=len(c), t_backbone=t_bb, t_align=0., timing="fallback_lt3_points")
            print(f"FALLBACK id {i} {pat}{n}{tag}: {len(c)} points", flush=True); continue
        lo, hi = (.1, ds.dmax) if A.dataset != "uni" else (max(1e-3, .25 * obs.min()), 4 * obs.max()); LO, HI = np.log(lo), np.log(hi)
        seed = int(hashlib.md5(f"{i}{pat}{n}{nz}0".encode()).hexdigest()[:7], 16); bases = {}
        dist = None
        memo = {}; DROP = {}
        def compute(v):
            """(base, [mean, median] full-resolution residual maps, seconds, context rows used, calls, queries, t_align) for variant v."""
            if v in memo: return memo[v]
            cf = VARS[v]; s = time.perf_counter()
            if "ens" in cf:
                ms = [compute(m if m in VARS else "_" + m) for m in cf["ens"]]
                assert all(np.array_equal(m[0], ms[0][0]) for m in ms), "ensemble members must share the aligned map"
                memo[v] = (ms[0][0], [np.mean([m[1][j] for m in ms], 0) for j in (0, 1)], sum(m[2] for m in ms), max(m[3] for m in ms), sum(m[4] for m in ms), ms[0][5], ms[0][6], None); return memo[v]
            if cf["prior"] == "none":
                base = np.full(H * W, float(np.exp(np.median(np.log(obs))))); base_c = base[c]; t_alv = 0.; cols = [COORD]
            else:
                pr = cf["prior"]
                if pr == "ma":   # indoor vs outdoor metric model: the one whose log-ratio to the points is less dispersed
                    dev_ = [np.median(np.abs((np.log(obs) - F_["mprior"][c, j]) - np.median(np.log(obs) - F_["mprior"][c, j]))) for j in (0, 1)]; pr = "mi" if dev_[0] <= dev_[1] else "mo"
                lm = None if pr == "rel" else F_["mprior"][:, 0 if pr == "mi" else 1].astype(np.float64)   # log metric depth
                pv = rel if pr == "rel" else np.exp(-lm)   # inverse metric depth plays the role of rel
                gk = "g" + pr + ("" if cf["align"] in ("global", "local") else cf["align"]); key = gk if cf["align"] != "local" else f"{gk}local{cf['k']}"
                if gk not in bases:
                    sa = time.perf_counter()
                    gb = np.clip(np.exp(lm + np.median(np.log(obs) - lm[c])), lo, hi) if cf["align"] == "scale" else align(pv, c, obs, lo, hi, {"wlog": obs, "wdep": obs ** 2}.get(cf["align"]))
                    bases[gk] = (gb, gb[c], time.perf_counter() - sa)
                if key not in bases:
                    sa = time.perf_counter(); lb, lbc = local_ss(bases[gk][0], c, obs, COORD, cf["k"], lo, hi); bases[key] = (lb, lbc, bases[gk][2] + time.perf_counter() - sa)
                base, base_c, t_alv = bases[key]; cols = [COORD, pv.astype(np.float32)]
            y = (obs - base_c) if cf["target"] == "lin" else (np.log(obs) - np.log(base_c))
            X = np.column_stack(cols + [F_[f] for f in cf["feats"] if f != "nn"]).astype(np.float32)
            if "nn" in cf["feats"] and len(c) >= 6:
                tr = cKDTree(COORD[c]); kq = 4; d1, n1 = tr.query(COORD, k=kq); d2, n2 = tr.query(COORD[c], k=kq + 1); d1 = d1.copy(); n1 = n1.copy(); d1[c] = d2[:, 1:]; n1[c] = n2[:, 1:]
                w = 1 / (d1 + 1e-3); yn = y[n1]; X = np.column_stack([X, (w * yn).sum(1) / w.sum(1), np.log(d1[:, 0] + 1e-3), yn.std(1)]).astype(np.float32)
            elif "nn" in cf["feats"]: X = np.column_stack([X, np.zeros((len(X), 3), np.float32)])
            if cf["stride"] == 2:
                H2, W2 = H // 2, W // 2; g = torch.from_numpy(X.reshape(H, W, -1)).permute(2, 0, 1)[None].float()
                Qall = F.avg_pool2d(g, 2)[0].permute(1, 2, 0).reshape(-1, X.shape[1]).numpy(); sel = np.flatnonzero(need_cells(valid, H, W, H2, W2))
            else: Qall = X; sel = np.flatnonzero(valid)
            Q = Qall[sel]; rng = np.random.default_rng(seed)
            cc, yv = c, y; n_drop = 0
            if cf.get("xf") and len(c) >= 40:
                # cross-fitted consistency filter: each half of the measurements is predicted from the other half; a measurement whose label is further than xf robust
                # standard deviations (MAD) from its cross-fitted prediction is dropped from the context. Uses only the input points, never ground truth.
                pm = rng.permutation(len(c)); hv = [pm[0::2], pm[1::2]]; res = np.zeros(len(c)); wid = np.full(len(c), np.nan)
                for a_, b_ in ((0, 1), (1, 0)):
                    ka = hv[a_] if len(hv[a_]) <= cf["cap"] else rng.choice(hv[a_], cf["cap"], replace=False)
                    o_ = tfm_predict(cf["tfm"], X[c[ka]], y[ka], X[c[hv[b_]]], seed + 7 + a_, cf["nest"]); res[hv[b_]] = y[hv[b_]] - o_[1]
                    if len(o_) == 4: wid[hv[b_]] = (o_[3] - o_[2]) / 3.29   # the model's own predictive spread (90% interval as a standard deviation)
                dv = np.abs(res - np.median(res)); sg = max(1.4826 * np.median(dv), cf.get("xf_floor", .01))
                if cf.get("xf_mode") == "q" and np.isfinite(wid).all(): sg = np.maximum(wid, cf.get("xf_floor", .01))   # per-point: wide where the model is unsure (depth edges), so true edge points survive
                keep = dv <= np.maximum(cf["xf"] * sg, cf.get("xf_abs", 0.)); n_drop = int((~keep).sum())
                if keep.sum() >= 20: cc, yv = c[keep], y[keep]
            if cf.get("ctx") == "local":
                # one context per query tile: the nloc points nearest to the tile (random tie-break inside it) plus nglob random others; tiles overlap and are blended with tent weights
                ny, nx = cf["tiles"]; qy, qx = Qall[sel, 0] * (H - 1), Qall[sel, 1] * (W - 1); cy, cx = cc // W, cc % W; acc = np.zeros((2, len(sel))); wsum = np.zeros(len(sel)); used_ = set(); ncalls = 0
                ey, ex = np.linspace(0, H, ny + 1), np.linspace(0, W, nx + 1); oy, ox = .25 * H / ny, .25 * W / nx
                for a_ in range(ny):
                    for b_ in range(nx):
                        y0, y1, x0, x1 = ey[a_], ey[a_ + 1], ex[b_], ex[b_ + 1]; Y0, Y1, X0, X1 = y0 - oy, y1 + oy, x0 - ox, x1 + ox
                        qs = np.flatnonzero((qy >= Y0) & (qy < Y1) & (qx >= X0) & (qx < X1))
                        if len(qs) == 0: continue
                        d_ = np.hypot(np.maximum(0, np.maximum(y0 - cy, cy - y1)), np.maximum(0, np.maximum(x0 - cx, cx - x1))); od = np.argsort(d_ + rng.random(len(d_)) * 1e-3)
                        k_ = od[:cf["nloc"]]; rest = od[cf["nloc"]:]
                        if cf.get("nglob", 0) and len(rest): k_ = np.concatenate([k_, rng.choice(rest, min(cf["nglob"], len(rest)), replace=False)])
                        m_, md_ = tfm_predict(cf["tfm"], X[cc[k_]], yv[k_], Q[qs], seed + 1000 + ncalls, cf["nest"])[:2]; ncalls += 1; used_.update(k_.tolist())
                        w_ = np.maximum(np.minimum.reduce([qy[qs] - Y0, Y1 - qy[qs], qx[qs] - X0, X1 - qx[qs]]), 1e-3)
                        acc[0, qs] += w_ * m_; acc[1, qs] += w_ * md_; wsum[qs] += w_
                assert (wsum > 0).all(); parts = [(acc[0] / wsum, acc[1] / wsum)]; subs = [np.array(sorted(used_))] * max(ncalls, 1)
            else:
                if len(cc) > cf["cap"]:
                    # K balanced disjoint groups, each topped up to the full cap with random other points: no subset is ever tiny (e.g. N = 2006 gives two 2000-point subsets)
                    perm = rng.permutation(len(cc)); K = min(cf["K"], math.ceil(len(cc) / cf["cap"])); cap_ = cf["cap"]; subs = []
                    for g_ in np.array_split(perm[:K * cap_], K):
                        rest = np.setdiff1d(perm, g_, assume_unique=True); subs.append(np.concatenate([g_, rng.choice(rest, cap_ - len(g_), replace=False)]) if len(g_) < cap_ else g_)
                else: subs = [np.arange(len(cc))]
                parts = [tfm_predict(cf["tfm"], X[cc[sb]], yv[sb], Q, seed + (100 + b if len(subs) > 1 else 0), cf["nest"]) for b, sb in enumerate(subs)]
            rs = []
            for j in ((0, 1, 2, 3) if all(len(p_) == 4 for p_ in parts) and len(parts) == 1 else (0, 1)):   # quantiles kept only for a single-context Kumo call
                r = np.zeros(len(Qall), np.float32); r[sel] = np.mean([p[j] for p in parts], 0)
                if cf["stride"] == 2: r = F.interpolate(torch.from_numpy(r).reshape(1, 1, H2, W2), size=(H, W), mode="bilinear", align_corners=False).numpy().reshape(-1)
                if cf["gate"]: r = r / (1 + (cKDTree(COORD[c]).query(COORD)[0] / cf["gate"]) ** 2)
                rs.append(r)
            torch.cuda.synchronize()
            DROP[v] = n_drop
            memo[v] = (base, rs, time.perf_counter() - s + sum(t_f[f] for f in cf["feats"]), int(len(subs[0]) if cf.get("ctx") == "local" else sum(len(sb) for sb in subs)), len(subs), len(sel), t_alv, rs[2:] if len(rs) == 4 else None); rs = rs[:2]; memo[v] = memo[v][:1] + (rs,) + memo[v][2:]; return memo[v]
        for v in todo:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            base, rs, sec, used, ncalls, nq, t_alv, rq = compute(v); cf = VARS[v]; out = [np.clip(base + r, lo, hi) if cf.get("target") == "lin" else np.exp(np.clip(np.log(base) + r, LO, HI)) for r in rs]
            peak = torch.cuda.max_memory_allocated() / 2 ** 30
            for st, p in zip(("mean", "med"), out):
                wr.add(i, pat, n, "large", f"{v}/{st}{tag}", p, t, c_given, sec, peak, retained_ctx=min(len(c), cf.get("cap", 0) or len(c)), n_ctx=used, t_backbone=t_bb, t_align=t_alv,
                       timing=f"correction_only;calls={ncalls};queries={nq};HxW={H}x{W};pts={len(c)};dropped={DROP.get(v, 0)}")
            if A.diag:
                if dpx is None: dpx = cKDTree(np.column_stack([c_given // W, c_given % W])).query(np.column_stack([np.arange(H * W) // W, np.arange(H * W) % W]))[0]
                e = out[1] - t; vv = valid
                for kind_, val_, edges_ in (("dist", dpx, DEDGE), ("depth", t, ZEDGE)):
                    b_ = np.digitize(val_[vv], edges_)
                    for bb in np.unique(b_):
                        mk = b_ == bb; DG.writerow([i, pat, n, f"{v}{tag}", kind_, edges_[bb - 1] if bb > 0 else -1, int(mk.sum()), float((e[vv][mk] ** 2).sum()), float(np.abs(e[vv][mk]).sum()), ""]); 
                if rq is not None and cf.get("target") != "lin":
                    un = vv.copy(); un[c_given] = False; lo_, hi_ = np.log(base) + rq[0], np.log(base) + rq[1]; lt = np.log(np.maximum(t, 1e-6))
                    DG.writerow([i, pat, n, f"{v}{tag}", "cov90", 90, int(un.sum()), float(((lt >= lo_) & (lt <= hi_))[un].sum()), float((hi_ - lo_)[un].sum()), ""])
                DGF.flush()
            if SAVE_THIS: saved[f"{v}{tag}|{pat}{n}"] = out[1].astype(np.float16)
    if SAVE_THIS and saved:   # feature maps for the method figure
        saved["feat|rel"] = rel.astype(np.float32); saved["feat|pca8"] = pca8[:, :3].astype(np.float16)
        if "dino16" in F_: saved["feat|dino16"] = F_["dino16"][:, :6].astype(np.float16)
    if SAVE_THIS and saved: np.savez_compressed(A.out.replace(".csv", f".pred{i:05d}.npz"), gt=t.astype(np.float32), **{k_.replace("/", "_"): v_ for k_, v_ in saved.items()})
    print(f"ours image {kk+1} id {i} {H}x{W} {time.time()-t0:.0f}s", flush=True)
print("OURS DONE", flush=True)
