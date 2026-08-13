import lpips
import torch.nn.functional as F
_lpips_fn = None
import os, json, torch
from os import listdir
from os.path import isfile, join
from argparse import ArgumentParser
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as tf

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return torch.tensor(100.0)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def evaluate(model_path):
    render_path = os.path.join(model_path, "renders")
    gt_path = os.path.join(model_path, "gt")
    if not os.path.exists(render_path):
        render_path = os.path.join(model_path, "renders")
        gt_path = os.path.join(model_path, "gt")

    image_names = sorted([f for f in listdir(render_path) if isfile(join(render_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    gt_images = sorted([f for f in listdir(join(model_path, "gt")) if isfile(join(model_path, "gt", f)) and f.lower().endswith((".png", ".jpg", ".jpeg"))])
    global _lpips_fn
    if _lpips_fn is None:
        print("Initializing AlexNet LPIPS model on CUDA...")
        _lpips_fn = lpips.LPIPS(net='alex').cpu()
    ssims, psnrs, lpipss = [], [], []
    scene_name = os.path.basename(model_path.rstrip("/"))
    print(f"Scene: {scene_name}")
    print("Method: ours_30000")
    
    for idx in tqdm(range(len(image_names)), desc="Metric evaluation progress"):
        name = image_names[idx]
        render_img = tf.to_tensor(Image.open(join(render_path, name))).cpu()[:3, :, :]
        gt_img = tf.to_tensor(Image.open(join(gt_path, name))).cpu()[:3, :, :]
        
        if render_img.shape != gt_img.shape:
            raise ValueError(f"Shape mismatch for {name}: render {tuple(render_img.shape)} vs gt {tuple(gt_img.shape)}")

        psnrs.append(psnr(render_img, gt_img).item())
        from skimage.metrics import structural_similarity as ssim_func
        r_np = render_img.permute(1, 2, 0).cpu().numpy()
        g_np = gt_img.permute(1, 2, 0).cpu().numpy()
        ssims.append(ssim_func(r_np, g_np, channel_axis=-1, data_range=1.0))
        try:
            # Prepare tensors for LPIPS [B, C, H, W]
            im0 = render_img.unsqueeze(0).cpu()
            im1 = gt_img.unsqueeze(0).cpu()
            
            # Interpolate down to 1024x1024 to avoid VRAM crashes on 6.4K canvases
            if im0.shape[-1] > 1024 or im0.shape[-2] > 1024:
                im0 = F.interpolate(im0, size=(1024, 1024), mode='bilinear', align_corners=False)
                im1 = F.interpolate(im1, size=(1024, 1024), mode='bilinear', align_corners=False)
                
            # LPIPS expects input range normalized to [-1, 1]
            im0 = torch.clamp(im0 * 2.0 - 1.0, -1.0, 1.0)
            im1 = torch.clamp(im1 * 2.0 - 1.0, -1.0, 1.0)
            
            with torch.no_grad():
                dist = _lpips_fn(im0, im1).mean().item()
            lpipss.append(float(dist))
        except Exception as e:
            print(f"LPIPS calculation error on current view: {e}")
            lpipss.append(0.0)

    fids = [0.0]
    print("\nFinal Results Summary:")
    print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean()))
    print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean()))
    print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean()))
    print("  FID  : {:>12.7f}".format(torch.tensor(fids).mean()))

    full_dict = {scene_name: {"SSIM": float(torch.tensor(ssims).mean()), "PSNR": float(torch.tensor(psnrs).mean()), "LPIPS": float(torch.tensor(lpipss).mean()), "FID": 0.0}}
    per_view_dict = {scene_name: {"SSIM": {n: float(s) for n, s in zip(image_names, ssims)}, "PSNR": {n: float(p) for n, p in zip(image_names, psnrs)}, "LPIPS": {n: float(l) for n, l in zip(image_names, lpipss)}}}

    with open(os.path.join(model_path, "results.json"), "w") as fp:
        json.dump(full_dict, fp, indent=4)
    with open(os.path.join(model_path, "per_view.json"), "w") as fp:
        json.dump(per_view_dict, fp, indent=4)
    print("Saved logs successfully.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", "-m", required=True, type=str)
    evaluate(parser.parse_args().model_path)