"""Supplementary qualitative figure: images at a fixed stride of the test order (not selected by result).   make_supp_figs.py NYU_BENCH_NPZ PRED_DIR OUT_DIR"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figs as M
ids = sorted(M.P)[2::4][:8]
M.qualitative([(i, "random500", f"#{i}") for i in ids], "supp_qual500.pdf")
M.qualitative([(i, "random50", f"#{i}") for i in ids], "supp_qual50.pdf")
print(ids)
