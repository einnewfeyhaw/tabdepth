"""Cross-image in-context depth with a BANK of context images: labelled rows come from many images of other scenes; the query image has no depth measurement.
NYUv2 test images (304x228). A fixed bank of 64 images, one from each of 64 randomly chosen scenes; queries are images of the remaining scenes.
Settings (pattern): bank4, bank16, bank64 = the first 4 / 16 / 64 bank images, 300 random labelled pixels from each (1,200 / 4,800 / 19,200 context rows).
Column sets (all frozen, PCA bases fitted on the bank images only and applied to the query):
  dino     position, RGB, DINOv2-L token PCA(32)
  dino_ms  dino + PCA(16) of tokens averaged over a 5x5 token window + PCA(16) of the image-mean token (a feature pyramid: pixel, neighbourhood, whole image)
  prior    position, standardised DA-V2 relative depth with gradients and smoothings, DA-V2 token PCA(8)
Regressors on identical rows: tabicl (all rows), kumo (2,000 random rows), ridge (linear probe, alpha by cross-validation), gbt2. Label = log depth.
Baselines: metric = DA-V2 metric-indoor network; affine = inverse-depth scale/shift fitted on the bank and applied to the query's DA-V2 map; const = bank median."""
import argparse, time, csv, warnings
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from scipy.ndimage import gaussian_filter, sobel
from scipy.optimize import lsq_linear
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModelForDepthEstimation, AutoModel
import sdm
from tabicl import TabICLRegressor
from common import get_dataset, Writer, shard
warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
for a, d in [("--data", None), ("--out", None), ("--scenes", "nyu_scenes.csv")]: ap.add_argument(a, default=d)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--every", type=int, default=4)
A = ap.parse_args(); DEV = "cuda"; ds = get_dataset("nyu", A.data); H, W = ds.H, ds.W; wr = Writer(A.out)
LM = "depth-anything/Depth-Anything-V2-Large-hf"; proc, dm = AutoImageProcessor.from_pretrained(LM), AutoModelForDepthEstimation.from_pretrained(LM).to(DEV).eval()
MM = "depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf"; mproc, mm = AutoImageProcessor.from_pretrained(MM), AutoModelForDepthEstimation.from_pretrained(MM).to(DEV).eval()
DINO = AutoModel.from_pretrained("facebook/dinov2-large").to(DEV).eval(); kumo = sdm.models.KumoTabular(task="regression", size="large", device=DEV).eval()
yy, xx = np.mgrid[0:H, 0:W]; COORD = np.stack([yy / (H - 1), xx / (W - 1)], -1).reshape(-1, 2).astype(np.float32)
scene = {int(r["image"]): r["scene"] for r in csv.DictReader(open(A.scenes))}
scn = sorted(set(scene.values())); R0 = np.random.default_rng(0); bank_sc = [scn[k] for k in R0.permutation(len(scn))[:128]]
# one image per scene first, then a second image of each scene
BANK = [min(k for k, s in scene.items() if s == b) for b in bank_sc]; BANK += [m for m in (max(k for k, s in scene.items() if s == b) for b in bank_sc) if m not in BANK]; QUERIES = [k for k in sorted(scene) if scene[k] not in set(bank_sc)][::A.every]

def feats(i):
    img = ds.rgb(i)
    with torch.inference_mode():
        inp = proc(images=Image.fromarray(img), return_tensors="pt").to(DEV); o = dm(**inp, output_hidden_states=True)
        rel = F.interpolate(o.predicted_depth[:, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy().reshape(-1)
        gh, gw = inp.pixel_values.shape[-2] // 14, inp.pixel_values.shape[-1] // 14; hs = o.hidden_states[-1][0]; tk = hs[hs.shape[0] - gh * gw:].float().reshape(gh, gw, -1)
        sc = 518 / min(H, W); h, w = int(round(H * sc / 14)) * 14, int(round(W * sc / 14)) * 14
        x = F.interpolate(torch.from_numpy(img).permute(2, 0, 1)[None].float().to(DEV) / 255, size=(h, w), mode="bicubic", align_corners=False)
        x = (x - torch.tensor([.485, .456, .406], device=DEV).view(1, 3, 1, 1)) / torch.tensor([.229, .224, .225], device=DEV).view(1, 3, 1, 1)
        dk = DINO(pixel_values=x).last_hidden_state[0, 1:].float().reshape(h // 14, w // 14, -1)
        md = F.interpolate(mm(**mproc(images=Image.fromarray(img), return_tensors="pt").to(DEV)).predicted_depth[:, None], size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy().reshape(-1)
    rz = (rel - rel.mean()) / (rel.std() + 1e-6); R2 = rz.reshape(H, W); gx, gy = sobel(R2, 1) / 8, sobel(R2, 0) / 8
    geo = np.column_stack([rz, gx.reshape(-1), gy.reshape(-1), np.hypot(gx, gy).reshape(-1), gaussian_filter(R2, 4).reshape(-1), gaussian_filter(R2, 12).reshape(-1)]).astype(np.float32)
    return dict(rgb=(img.reshape(-1, 3) / 255.).astype(np.float32), rz=rz, geo=geo, tk=tk, dk=dk, md=md, gt=ds.gt(i))

def basis(toks, q):
    torch.manual_seed(0); allt = torch.cat([t.reshape(-1, t.shape[-1]) for t in toks]); mu = allt.mean(0); _, _, V = torch.pca_lowrank(allt - mu, q=q, center=False); return mu, V[:, :q]
def proj(t, mu, V, q):   # (gh, gw, C) tokens -> (H*W, q) map
    c = ((t - mu) @ V[:, :q]).permute(2, 0, 1)[None]; return F.interpolate(c, size=(H, W), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).reshape(-1, q).cpu().numpy().astype(np.float32)
def pooled(t):
    return F.avg_pool2d(t.permute(2, 0, 1)[None], 5, stride=1, padding=2, count_include_pad=False)[0].permute(1, 2, 0)
def columns(f, PB):
    (mu, V), (mu2, V2) = PB; d32 = proj(f["dk"], mu, V, 32); p16 = proj(pooled(f["dk"]), mu, V, 16); g16 = np.repeat(((f["dk"].reshape(-1, f["dk"].shape[-1]).mean(0) - mu) @ V[:, :16]).cpu().numpy()[None].astype(np.float32), H * W, 0)
    dino = np.column_stack([COORD, f["rgb"], d32]); return {"dino": dino, "dino_ms": np.column_stack([dino, p16, g16]), "prior": np.column_stack([COORD, f["geo"], proj(f["tk"], mu2, V2, 8)])}

def predict(kind, X, y, Q, seed):
    s = StandardScaler().fit(X); Xs, Qs = s.transform(X).astype(np.float32), s.transform(Q).astype(np.float32)
    if kind == "ridge":
        from sklearn.linear_model import RidgeCV; return RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(Xs, y).predict(Qs)
    if kind == "gbt2":
        from sklearn.ensemble import HistGradientBoostingRegressor; return HistGradientBoostingRegressor(max_iter=400, learning_rate=.05, min_samples_leaf=5, l2_regularization=1.0, random_state=seed).fit(Xs, y).predict(Qs)
    if kind == "tabicl":
        r = TabICLRegressor(n_estimators=4, random_state=seed, device=DEV); r.fit(Xs, y.astype(np.float32))
        try: return np.asarray(r.predict(Qs, output_type="median")).reshape(-1)
        except TypeError: return np.asarray(r.predict(Qs)).reshape(-1)
    K = int(kind[4:]); rg = np.random.default_rng(seed); outs = []
    for _ in range(K):   # K random subsets of 2,000 context rows, predictions averaged
        k = rg.choice(len(Xs), min(2000, len(Xs)), replace=False); outs.append(kumo_once(Xs[k], y[k], Qs, seed))
    return np.mean(outs, 0)
def kumo_once(Xs, y, Qs, seed):
    T = sdm.TableTensor; nm = [f"x{j}" for j in range(Xs.shape[1])]; tb = lambda t, n_: T(columns={"numerical": n_}, numerical=t); med = []; gen = torch.Generator(device=DEV).manual_seed(seed)
    try:
        with torch.inference_mode():
            kumo.fit(x=tb(torch.as_tensor(Xs, device=DEV), nm), y=tb(torch.as_tensor(y, dtype=torch.float32, device=DEV).reshape(-1, 1), ["y"]), num_estimators=2, estimator_batch_size=1, generator=gen)
            for j in range(0, len(Qs), 4096):
                q = kumo.predict(tb(torch.as_tensor(Qs[j:j + 4096], device=DEV), nm)).numerical
                if q.ndim == 3 and q.shape[0] == 1: q = q[0]
                med.append(q.float()[:, 499].cpu().numpy())
    finally: kumo.clear()
    return np.concatenate(med)
def half(X):
    g = torch.from_numpy(X.reshape(H, W, -1)).permute(2, 0, 1)[None].float(); return F.avg_pool2d(g, 2)[0].permute(1, 2, 0).reshape(-1, X.shape[1]).numpy()
def up(r): return F.interpolate(torch.from_numpy(np.asarray(r, dtype=np.float32)).reshape(1, 1, H // 2, W // 2), size=(H, W), mode="bilinear", align_corners=False).numpy().reshape(-1)

t0 = time.time(); BF = [feats(j) for j in BANK]; print(f"bank features {time.time()-t0:.0f}s; bank {len(BANK)} images, {len(QUERIES)} queries", flush=True)
CTX = {}
SIZES = (64, 128, len(BANK))
for nb in SIZES:
    fb = BF[:nb]; PB = (basis([f["dk"] for f in fb], 32), basis([f["tk"] for f in fb], 8)); rows = {k: [] for k in ("dino", "dino_ms", "prior")}; ys, rzs = [], []
    for b, f in enumerate(fb):
        v = np.flatnonzero(f["gt"] > 0); p = np.random.default_rng(100 + b).choice(v, min(300, len(v)), replace=False); cs = columns(f, PB)
        for k in rows: rows[k].append(cs[k][p])
        ys.append(np.log(f["gt"][p])); rzs.append(f["rz"][p])
    CTX[nb] = (PB, {k: np.concatenate(v).astype(np.float32) for k, v in rows.items()}, np.concatenate(ys), np.concatenate(rzs))
for f in BF: f.pop("dk"); f.pop("tk")
torch.cuda.empty_cache()
for kk, pos in enumerate(shard(len(QUERIES), A.shard, A.nshard, A.limit)):
    i = QUERIES[pos]; fq = feats(i); t = fq["gt"]
    for nb in SIZES:
        name = f"bank{nb}"; PB, XC, y, rzc = CTX[nb]; n = len(y)
        if wr.has(i, name, n, "large", "const"): continue
        cq = columns(fq, PB)
        for cs in ("dino_ms",):
            Q = half(cq[cs])
            for kind in ("kumo1", "kumo4", "kumo16", "ridge", "gbt2") + (("tabicl",) if n <= 40000 else ()):
                s = time.perf_counter(); pred = np.exp(np.clip(up(predict(kind, XC[cs], y, Q, 1000 + i)), np.log(.1), np.log(10.)))
                wr.add(i, name, n, "large", f"{cs}/{kind}", pred, t, np.array([], dtype=int), time.perf_counter() - s, 0., timing=f"bank;images={nb}")
        co = lsq_linear(np.column_stack([rzc, np.ones(n)]), 1 / np.exp(y), bounds=([0, -np.inf], [np.inf, np.inf])).x; aff = np.clip(1 / np.maximum(co[0] * fq["rz"] + co[1], 1e-3), .1, 10.)
        for nm_, p_ in (("affine", aff), ("metric", np.clip(fq["md"], .1, 10.)), ("const", np.full(H * W, float(np.exp(np.median(y)))))):
            wr.add(i, name, n, "large", nm_, p_, t, np.array([], dtype=int), 0., 0., timing=f"bank;images={nb}")
    print(f"bank query {kk+1} id {i} {time.time()-t0:.0f}s", flush=True)
print("BANK DONE", flush=True)
