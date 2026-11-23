"""Prior Depth Anything, official interface. --frozen picks the coarse-stage Depth Anything V2 size (default vitb = official default;
vits / vitl match our small / large backbones). Conditioned model is the released ViT-B. Full and official coarse-only outputs."""
import argparse, os, tempfile, time
import numpy as np, torch
from prior_depth_anything import PriorDepthAnything
from common import get_dataset, parse_conds, Writer, shard, sha256, provenance
import glob as _g
ap = argparse.ArgumentParser()
for a, d in [("--dataset", "nyu"), ("--data", None), ("--ctx", None), ("--out", None), ("--frozen", "vitb"), ("--conds", ""), ("--noise", "clean"), ("--seeds", "0")]: ap.add_argument(a, default=d)
ap.add_argument("--every", type=int, default=1)
ap.add_argument("--idfile", default="")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
A = ap.parse_args()
ds = get_dataset(A.dataset, A.data, A.ctx); wr = Writer(A.out); tmp = tempfile.mkdtemp(); conds = parse_conds(A.conds, ds.conds)
tag = "" if A.frozen == "vitb" else "_" + A.frozen
for name, coarse in (("priorda" + tag, False), ("priorda_coarse" + tag, True)):
    m = PriorDepthAnything(device="cuda:0", coarse_only=coarse, frozen_model_size=A.frozen)
    if not coarse:
        fs = sorted(_g.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Rain729--Prior-Depth-Anything/snapshots/*/*.pth")))
        provenance(A.out, dict(repo="Rain729/Prior-Depth-Anything", version="1.1", frozen=A.frozen, conditioned="vitb",
                               files={os.path.basename(f): sha256(f) for f in fs}, input="official infer_one_sample on the full-resolution frame (518 internal)"))
    NZ = A.noise.split(","); SD = [int(x) for x in A.seeds.split(",")]; ids = [int(x) for x in open(A.idfile).read().split()] if A.idfile else list(range(0, ds.n_images, A.every))
    for k, pos in enumerate(shard(len(ids), A.shard, A.nshard, A.limit)):
        i = ids[pos]; ip = os.path.join(tmp, "rgb.npy"); np.save(ip, ds.rgb(i)); t = ds.gt(i)
        for pat, n in conds:
          for nz in NZ:
           for sd in (SD if nz == "clean" else [0]):
            nm = name + ("" if nz == "clean" and sd == 0 else "@" + nz + ("#s%d" % sd if sd else ""))
            if wr.has(i, pat, n, "own", nm): continue
            c, obs = ds.sample(i, pat, n, noise=nz, seed=sd); sp = np.zeros(ds.H * ds.W, np.float32); sp[c] = obs
            pp = os.path.join(tmp, "prior.npy"); np.save(pp, sp.reshape(ds.H, ds.W))
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); s = time.perf_counter()
            with torch.no_grad(): p = m.infer_one_sample(image=ip, prior=pp, visualize=False)
            torch.cuda.synchronize(); sec = time.perf_counter() - s
            p = p.squeeze().float().cpu().numpy(); assert p.shape == (ds.H, ds.W), p.shape
            wr.add(i, pat, n, "own", nm, p, t, c, sec, torch.cuda.max_memory_allocated() / 2 ** 30, retained_ctx=len(c), timing="end_to_end")
        if k % 20 == 0: print(name, k, flush=True)
    del m; torch.cuda.empty_cache()
print("PRIORDA DONE", flush=True)
