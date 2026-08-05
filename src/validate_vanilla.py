import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from utils.loss_utils import ssim
from lpips import LPIPS

def load_image(path, target_size=None):
    img = Image.open(path).convert('RGB')
    if target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_tensor = F.to_tensor(img).unsqueeze(0).cuda()
    return img_tensor

def evaluate_vanilla_baseline(render_dir, gt_dir):
    print("Initializing AlexNet LPIPS model on CUDA...")
    lpips_metric = LPIPS(net='alex').cuda()
    
    if not os.path.exists(render_dir):
        print(f"Error: Render directory does not exist: {render_dir}")
        return
        
    render_files = sorted([f for f in os.listdir(render_dir) if f.endswith('.png')])
    if not render_files:
        print(f"Error: No .png images found in {render_dir}")
        return
        
    psnrs, ssims, lpips_vals = [], [], []
    
    print(f"Evaluating metrics over {len(render_files)} frames...")
    for f in render_files:
        render_path = os.path.join(render_dir, f)
        gt_path = os.path.join(gt_dir, f)
        
        if not os.path.exists(gt_path):
            continue
            
        render_img = load_image(render_path)
        _, _, h, w = render_img.shape
        
        gt_img = load_image(gt_path, target_size=(w, h))
        
        mse = torch.mean((render_img - gt_img) ** 2).item()
        psnr = 100 if mse == 0 else 20 * np.log10(1.0 / np.sqrt(mse))
        ssim_val = ssim(render_img, gt_img).item()
        lpips_val = lpips_metric(render_img * 2.0 - 1.0, gt_img * 2.0 - 1.0).item()
        
        psnrs.append(psnr)
        ssims.append(ssim_val)
        lpips_vals.append(lpips_val)
        
    print("\n=== Final Vanilla 2x Baseline Results Summary ===")
    print(f"  PSNR : {np.mean(psnrs):.7f}")
    print(f"  SSIM : {np.mean(ssims):.7f}")
    print(f"  LPIPS: {np.mean(lpips_vals):.7f}")

if __name__ == "__main__":
    RENDER_DIR = "output/progressive_run/test/ours_30000/renders"
    GT_DIR = "data/tandt/Museum/images" 
    
    evaluate_vanilla_baseline(RENDER_DIR, GT_DIR)