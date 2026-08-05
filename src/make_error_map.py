import os
import numpy as np
from PIL import Image

img_path = '/home/sri/SplatSuRe/output/now_8x/train/ours_7000/renders/00023.png'
gt_path = '/home/sri/SplatSuRe/output/now_8x/train/ours_7000/gt/00023.png'
out_path = '/home/sri/SplatSuRe/error_map_00023.png'

if not os.path.exists(img_path) or not os.path.exists(gt_path):
    print("Error: Could not find render or GT file. Check paths.")
else:
    # Load images as grayscale numpy arrays
    img = np.array(Image.open(img_path).convert('L'), dtype=np.float32)
    gt = np.array(Image.open(gt_path).convert('L'), dtype=np.float32)
    
    # Compute absolute pixel-wise difference
    diff = np.abs(img - gt)
    
    # Maximize contrast so errors are brightly visible (normalize to 0-255)
    if diff.max() > 0:
        diff = (diff / diff.max()) * 255.0
    
    # Save the error map
    Image.fromarray(diff.astype(np.uint8)).save(out_path)
    print(f"Success! Error map saved to {out_path}")
