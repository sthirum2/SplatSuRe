import torch
import numpy as np
import cv2
import sys

try:
    # 1. Load the weight map natively using PyTorch (handles the zip/pickle wrapping automatically)
    weight_tensor = torch.load(sys.argv[1], map_location="cpu")
    
    # 2. Convert PyTorch tensor to a NumPy array
    if isinstance(weight_tensor, torch.Tensor):
        weight_map = weight_tensor.detach().numpy()
    else:
        weight_map = np.asarray(weight_tensor)
        
    print(f"Successfully loaded weight map!")
    print(f"Shape: {weight_map.shape} | Dtype: {weight_map.dtype}")
    print(f"Value range: Min = {weight_map.min():.4f}, Max = {weight_map.max():.4f}, Mean = {weight_map.mean():.4f}")
    
    # 3. Normalize to 0-255 range for visual inspection
    if weight_map.max() - weight_map.min() > 1e-5:
        visual_map = (weight_map - weight_map.min()) / (weight_map.max() - weight_map.min())
    else:
        visual_map = np.zeros_like(weight_map)
    
    visual_map = (visual_map * 255).astype(np.uint8)
    
    # 4. Save as a visual PNG image
    cv2.imwrite("test_weight_map.png", visual_map)
    print("Saved visualization to test_weight_map.png")
    
except Exception as e:
    import traceback
    print(f"Error: {e}")
    traceback.print_exc()
