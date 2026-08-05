from pathlib import Path
import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from dreamsim import dreamsim

# Empty any fragmented CUDA allocation states before spinning up the heavy transformer
torch.cuda.empty_cache()

print("Loading DreamSim model onto CUDA...")
dreamsim_model, dreamsim_preprocess = dreamsim(pretrained=True, device='cuda')

def readImages(renders_dir, gt_dir, target_size=(512, 512)):
    renders = []
    gts = []
    image_names = []
    for fname in sorted(os.listdir(renders_dir)):
        if fname.endswith(('.png', '.jpg', '.jpeg')):
            # Fast, memory-light downsampling on load
            render = Image.open(renders_dir / fname).convert('RGB').resize(target_size, Image.Resampling.BILINEAR)
            gt = Image.open(gt_dir / fname).convert('RGB').resize(target_size, Image.Resampling.BILINEAR)
            
            renders.append(render)
            gts.append(gt)
            image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            test_dir = Path(scene_dir) / "test"

            if not test_dir.exists():
                continue

            for method in os.listdir(test_dir):
                method_dir = test_dir / method
                if not method_dir.is_dir():
                    continue
                
                print("Method:", method)
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                
                renders, gts, image_names = readImages(renders_dir, gt_dir, target_size=(512, 512))
                dreamsim_distances = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    # Process views cleanly one by one
                    render_ds = dreamsim_preprocess(renders[idx]).cuda()
                    gt_ds = dreamsim_preprocess(gts[idx]).cuda()
                    
                    with torch.no_grad():
                        dreamsim_distance = dreamsim_model(render_ds, gt_ds)
                    
                    dreamsim_distances.append(dreamsim_distance.item())
                    
                    # Clean up GPU context immediately per iteration
                    del render_ds, gt_ds
                    torch.cuda.empty_cache()

                mean_dreamsim = float(torch.tensor(dreamsim_distances).mean())
                print(f"\nFinal DreamSim Perceptual Metric: {mean_dreamsim:.7f}")

                results_path = Path(scene_dir) / "results.json"
                if results_path.exists():
                    with open(results_path, 'r') as fp:
                        res_data = json.load(fp)
                    
                    if method in res_data:
                        res_data[method]["LPIPS"] = mean_dreamsim  # Map to your target placeholder
                    else:
                        res_data[method] = {"DreamSim": mean_dreamsim}
                        
                    with open(results_path, 'w') as fp:
                        json.dump(res_data, fp, indent=4)
                    print("Updated metrics successfully in results.json.")

        except Exception as e:
            print(f"Error processing scene {scene_dir}: {e}")

if __name__ == "__main__":
    parser = ArgumentParser(description="DreamSim evaluation script")
    parser.add_argument("--model_paths", "-m", nargs="+", required=True)
    args = parser.parse_args()
    evaluate(args.model_paths)
