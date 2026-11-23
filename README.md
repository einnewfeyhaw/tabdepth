# TabDepth: Depth Completion as In-Context Tabular Regression

Code for the paper *Depth Completion as In-Context Tabular Regression*.

TabDepth treats depth completion as a table. Each pixel is a row, features from frozen image encoders are its columns, and the sparse depth measurements are its labels. A frozen tabular foundation model, trained only on synthetic tables, predicts the depth of every other pixel in context. Nothing is trained or fine-tuned.

Two variants:

- **TabDepth-D (direct):** columns are pixel position, colour and DINOv2 features. No component has been trained on depth.
- **TabDepth-P (prior-assisted):** a frozen Depth Anything V2 map is aligned to the measurements by scale and shift, and the tabular model predicts the log residual.

## Results

RMSE in metres on NYUv2 (654 test images, OMNI-DC evaluation data). OMNI-DC numbers are the published ones.

| Method | 500 pts | 200 | 100 | 50 | 5 |
|---|---|---|---|---|---|
| OMNI-DC | .111 | .147 | .180 | .225 | .536 |
| TabDepth-D (no depth-trained network) | .117 | .167 | .223 | .308 | .956 |
| TabDepth-P | **.108** | **.142** | **.169** | **.202** | **.357** |

TabDepth-P also has lower error than published OMNI-DC on VOID and on all 20 synthetic-pattern settings of the OMNI-DC benchmark (iBims-1, ARKitScenes, ETH3D, DIODE). OMNI-DC remains more accurate on real LiDAR (KITTI) and on structure-from-motion points (ETH3D); the paper analyses why.

## Layout

```
tabdepth/
  run_ours.py        the method, all variants and the diagnostics
  common.py          datasets, sampling of input points, metrics
  variants.json      named variants (see below)
  uni_prep.py        converts the OMNI-DC evaluation release to .npz
  prep.py, kitti_prep.py   NYUv2 / KITTI preparation for the diagnostics
  run_bank.py, run_bank2.py   context rows taken from other images (appendix)
  meas_audit.py      compares released input points with ground truth
  run_omnidc.py, run_priorda.py, run_marigold.py   baselines run on identical inputs
  analyze_uni.py, analyze_bench.py   tables from the result CSVs
  occ_setting.txt    the LiDAR occlusion pre-filter setting chosen on KITTI validation
splits/              image id lists (NYUv2 development images, KITTI validation subsets)
figures/             scripts that draw the paper figures
```

## Setup

A GPU is required. The runs in the paper used the package versions in `requirements.txt`.

```
pip install -r requirements.txt
```

Model weights are downloaded from the Hugging Face Hub on first use: Depth Anything V2 Large, DINOv2 Large, Kumo-Tabular (through `structured-data-models`), TabICLv2 and TabPFN v2. Some of them may require accepting a licence on the Hub.

## Data

The benchmark numbers use the evaluation data released with [OMNI-DC](https://github.com/princeton-vl/OMNI-DC), which fixes the images, the input points and the ground truth. Convert each setting to `.npz` files:

```
python tabdepth/uni_prep.py <omni-dc-eval-dir> <out-dir> <manifest.csv>
```

## Running

Run from inside `tabdepth/`. One call scores several variants on identical inputs and writes one CSV row per image and variant.

```
cd tabdepth

# TabDepth-P (kumo), TabDepth-D (nd_dino_rgb) and the aligned prior alone (zero) on one benchmark setting
python run_ours.py --dataset uni --data <out-dir>/NYU_test_500 \
    --variants kumo,nd_dino_rgb,zero --settings clean:NYU_test_500:0 --out results/ours_NYU_test_500.csv

# long contexts (KITTI): TabICLv2 with all measurements in one context
python run_ours.py --dataset uni --data <out-dir>/KITTIDC_test_LiDAR_64 \
    --variants icl --settings clean:KITTIDC_test_LiDAR_64:0 --out results/ours_KITTI_64.csv

# the same with the occlusion pre-filter (setting name from occ_setting.txt)
python run_ours.py --dataset uni --data <out-dir>/KITTIDC_test_LiDAR_64 \
    --variants icl --settings k2-3-10:KITTIDC_test_LiDAR_64:0 --out results/ours_KITTI_64_filtered.csv
```

Useful variant names in `variants.json`:

| Name | Meaning |
|---|---|
| `kumo` | TabDepth-P with Kumo-Tabular (default for up to 10,000 measurements) |
| `icl` | TabDepth-P with TabICLv2 (used above 10,000 measurements) |
| `nd_dino_rgb` | TabDepth-D |
| `zero` | aligned Depth Anything V2 only, no tabular model |
| `knn`, `gbt2`, `gp` | classical regressors on the same columns |

`--diag 1` also writes the error by distance to the nearest measurement and the coverage of the 90% predictive interval. `--shard i --nshard n` splits a setting over several GPUs. The docstring at the top of `run_ours.py` lists every option.

## Notes

This is research code, released as it was used for the paper. `variants.json` also contains variants that were tried and not adopted; the paper's appendix lists them.
