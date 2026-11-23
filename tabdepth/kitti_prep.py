"""Fixed nested LiDAR subsets for KITTI val_selection_cropped: for each frame, a random order of its valid LiDAR pixels (first 2000 kept).
Built once and shared so every method env reads identical indices."""
import sys, glob, os
import numpy as np
from PIL import Image
root, out = sys.argv[1], sys.argv[2]
d = os.path.join(root, "depth_selection", "val_selection_cropped")
imgs = sorted(glob.glob(os.path.join(d, "image", "*.png"))); assert len(imgs) == 1000
ctx, nl, ng = [], [], []
for k, p in enumerate(imgs):
    l = np.asarray(Image.open(p.replace("/image/", "/velodyne_raw/").replace("_sync_image_", "_sync_velodyne_raw_"))).reshape(-1)
    g = np.asarray(Image.open(p.replace("/image/", "/groundtruth_depth/").replace("_sync_image_", "_sync_groundtruth_depth_"))).reshape(-1)
    v = np.flatnonzero(l > 0); assert len(v) >= 2000, (p, len(v))
    ctx.append(np.random.default_rng(20_000 + k).permutation(v)[:2000]); nl.append(len(v)); ng.append(int((g > 0).sum()))
np.savez_compressed(out, ctx=np.stack(ctx).astype(np.int32), n_lidar=np.array(nl), n_gt=np.array(ng), names=np.array([os.path.basename(p) for p in imgs]))
print("kitti prep done; lidar points per frame: median %d min %d; GT pixels median %d" % (np.median(nl), min(nl), np.median(ng)))
