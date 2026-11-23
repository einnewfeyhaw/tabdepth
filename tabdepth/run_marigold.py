"""Marigold-DC, official pipeline defaults (50 steps, 768 px processing), ensemble 1. Repo must be on PYTHONPATH."""
import argparse, time
import numpy as np, torch
from PIL import Image
from diffusers import DDIMScheduler
from marigold_dc import MarigoldDepthCompletionPipeline
from common import get_dataset, parse_conds, Writer, shard, provenance
ap = argparse.ArgumentParser()
for a, d in [("--dataset", "nyu"), ("--data", None), ("--ctx", None), ("--out", None), ("--conds", "random:500,random:100,clustered:500")]: ap.add_argument(a, default=d)
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
A = ap.parse_args()
ds = get_dataset(A.dataset, A.data, A.ctx); wr = Writer(A.out); conds = parse_conds(A.conds, ds.conds)
pipe = MarigoldDepthCompletionPipeline.from_pretrained("prs-eth/marigold-depth-v1-0", prediction_type="depth").to("cuda")
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
pipe.set_progress_bar_config(disable=True)
provenance(A.out, dict(model="prs-eth/marigold-depth-v1-0", commit=getattr(pipe.unet.config, "_commit_hash", None), steps=50, ensemble=1, processing_resolution=768))
mine = shard(ds.n_images, A.shard, A.nshard, A.limit); t0 = time.time()
for k, i in enumerate(mine):
    t = ds.gt(i)
    for pat, n in conds:
        if wr.has(i, pat, n, "own", "marigold_dc_e1"): continue
        c, obs = ds.sample(i, pat, n); sp = np.zeros(ds.H * ds.W, np.float32); sp[c] = obs
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); s = time.perf_counter()
        p = pipe(image=Image.fromarray(ds.rgb(i)), sparse_depth=sp.reshape(ds.H, ds.W), num_inference_steps=50, ensemble_size=1, processing_resolution=768, seed=2024)
        torch.cuda.synchronize()
        wr.add(i, pat, n, "own", "marigold_dc_e1", np.asarray(p), t, c, time.perf_counter() - s, torch.cuda.max_memory_allocated() / 2 ** 30, retained_ctx=len(c), timing="end_to_end")
    print(f"image {k+1}/{len(mine)} id {i} {time.time()-t0:.0f}s", flush=True)
print("MARIGOLD DONE", flush=True)
