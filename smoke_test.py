"""Exercise every CLI stage using a tiny backbone and synthetic images, on CPU."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("--work",default="smoke_work")
a = p.parse_args()
root = Path(a.work).resolve()
root.mkdir(parents=True,exist_ok=True)
project = Path(__file__).parent.resolve()
env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
def run(*args):
    subprocess.run([sys.executable,*map(str,args)],cwd=project,env=env,check=True)
run("make_demo.py","--output",root/"data")
common = ["--config",project/"configs/smoke.json","--run",root/"run","--device","cpu"]
manifest = ["--manifest",root/"data/manifest.json"]
for cmd in ("validate-data","train-coarse","train-low","train-high"):
    run("-m","selective_seg.cli",cmd,*common,*manifest)
for split in ("selector","val"):
    run("-m","selective_seg.cli","cache",*common,*manifest,"--split",split)
run("-m","selective_seg.cli","train-selector",*common)
run("-m","selective_seg.cli","calibrate",*common)
run("-m","selective_seg.cli","evaluate",*common,*manifest,"--split","test",
    "--calibration",root/"run/calibration.json","--warmup","0","--passes","1")
print("SMOKE PASSED: training, caching, selector, calibration and test inference")
