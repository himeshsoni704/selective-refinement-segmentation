"""Generate tiny synthetic fixtures. These are NOT real benchmark data."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="demo_data")
args = parser.parse_args()
root = Path(args.output)
root.mkdir(parents=True,exist_ok=True)
rng = np.random.default_rng(42)
records = []
for i,split in enumerate(["train"]*3+["selector"]*2+["val"]*2+["test"]):
    y,x = np.mgrid[:79,:91]
    mask = (((x-(28+i*3))**2+(y-38)**2)<(14+i%3)**2).astype(np.uint8)
    image = np.stack((40+170*mask,110-65*mask,60+80*mask),-1).astype(np.float32)
    image = np.clip(image+rng.normal(0,10,image.shape),0,255).astype(np.uint8)
    Image.fromarray(image).save(root/f"image_{i}.png")
    Image.fromarray(mask).save(root/f"mask_{i}.png")
    records.append({"id":f"synthetic_{i}","image":f"image_{i}.png","mask":f"mask_{i}.png","split":split})
(root/"manifest.json").write_text(json.dumps(records,indent=2))
print(root/"manifest.json")
