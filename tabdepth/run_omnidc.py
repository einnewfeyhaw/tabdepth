"""OMNI-DC v1.1 settings from testing_scripts/demo.sh, weights from Hugging Face. Run from the repo's src/ directory."""
import sys, os, argparse, time, types
from unittest import mock
ap = argparse.ArgumentParser(); ap.add_argument("--data"); ap.add_argument("--out"); ap.add_argument("--bench"); ap.add_argument("--dataset", default="nyu"); ap.add_argument("--ctx"); ap.add_argument("--conds", default=""); ap.add_argument("--noise", default="clean"); ap.add_argument("--seeds", default="0"); ap.add_argument("--every", type=int, default=1); ap.add_argument("--idfile", default="")
ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
A = ap.parse_args(); sys.path.insert(0, A.bench)
sys.argv = ["x", "--max_depth", "300.0", "--data_normalize_median", "1", "--num_resolution", "3", "--multi_resolution_learnable_gradients_weights", "uniform",
            "--load_dav2", "1", "--gpus", "0", "--GRU_iters", "1", "--optim_layer_input_clamp", "1.0", "--depth_activation_format", "exp",
            "--whiten_sparse_depths", "1", "--gru_internal_whiten_method", "median", "--backbone_mode", "rgbd", "--pred_confidence_input", "1"]
sys.modules.setdefault("apex", mock.MagicMock())
import numpy as np, torch, torchvision.transforms as T
from PIL import Image
from config import args
_load = torch.load
import torchvision
# resnet34.pth only initialises layers; the Hugging Face checkpoint below overwrites them
torch.load = lambda f, *a, **k: torchvision.models.resnet34(pretrained=False).state_dict() if str(f).endswith("resnet34.pth") else _load(f, *a, **k)
from model.ognidc import OGNIDC
from common import get_dataset, parse_conds, Writer, shard, sha256, provenance
net = OGNIDC.from_pretrained("zuoym15/OMNI-DC", args=args).cuda().eval()
from huggingface_hub import snapshot_download
_snap = snapshot_download("zuoym15/OMNI-DC")
provenance(A.out, dict(repo="zuoym15/OMNI-DC", snapshot=_snap, files={f: sha256(os.path.join(_snap, f)) for f in os.listdir(_snap) if os.path.isfile(os.path.join(_snap, f))},
                       dav2_vitl_sha256=sha256("./depth_models/depth_anything_v2/checkpoints/depth_anything_v2_vitl.pth"), settings="testing_scripts/demo.sh v1.1"))
torch.load = _load
ds = get_dataset(A.dataset, A.data, A.ctx); wr = Writer(A.out); H, W = ds.H, ds.W; CONDS = parse_conds(A.conds, ds.conds)
tr = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
dv = int(4 * 2 ** (args.num_resolution - 1)); ph, pw = (-H) % dv, (-W) % dv
pad = torch.nn.functional.pad
NZ = A.noise.split(","); SD = [int(x) for x in A.seeds.split(",")]; ids = [int(x) for x in open(A.idfile).read().split()] if A.idfile else list(range(0, ds.n_images, A.every))
for k, pos in enumerate(shard(len(ids), A.shard, A.nshard, A.limit)):
    i = ids[pos]; x = tr(Image.fromarray(ds.rgb(i)))[None].cuda(); t = ds.gt(i)
    for pat, n in CONDS:
      for nz in NZ:
       for sd in (SD if nz == "clean" else [0]):
        name = "omni_dc" + ("" if nz == "clean" and sd == 0 else "@" + nz + ("#s%d" % sd if sd else ""))
        if wr.has(i, pat, n, "own", name): continue
        c, obs = ds.sample(i, pat, n, noise=nz, seed=sd); sp = np.zeros(H * W, np.float32); sp[c] = obs
        d = torch.from_numpy(sp.reshape(1, 1, H, W)).cuda(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); s = time.perf_counter()
        with torch.no_grad():
            o = net({"rgb": pad(x, (0, pw, 0, ph)), "dep": pad(d, (0, pw, 0, ph)), "K": torch.eye(3).reshape(1, 3, 3).cuda(), "pattern": 0})
        torch.cuda.synchronize(); sec = time.perf_counter() - s
        p = o["pred"][..., :H, :W].squeeze().cpu().numpy()
        wr.add(i, pat, n, "own", name, p, t, c, sec, torch.cuda.max_memory_allocated() / 2 ** 30, retained_ctx=len(c), timing="end_to_end")
    if k % 20 == 0: print("omni", k, flush=True)
print("OMNI DONE", flush=True)
