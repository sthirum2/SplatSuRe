import cv2
import numpy as np

def compute_error_map(img1_path, img2_path, output_name):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print(f"[-] Error loading: {img1_path} or {img2_path}")
        return False

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

    # Compute absolute pixel-level delta
    error_map = cv2.absdiff(img1, img2)
    
    # Apply Jet color map (Blue = Perfect fit, Red = High Error)
    color_error_map = cv2.applyColorMap(error_map, cv2.COLORMAP_JET)
    
    cv2.imwrite(output_name, color_error_map)
    print(f"[+] Saved: {output_name}")
    return True

# --- RUN PATH CONFIGURATIONS ---
# 1. Pranav's Diagnostic Ask: StableSR vs Ground Truth
compute_error_map(
    img1_path='data/tandt/Museum/images_SR/00023.png', 
    img2_path='data/tandt/Museum/images_2/00023.jpg',
    output_name='error_stablesr_vs_gt.png'
)

# 2. Your Validation Ask: Your Final Progressive Render vs Ground Truth
compute_error_map(
    img1_path='output/now_8x/train/ours_15000/renders/00023.png', 
    img2_path='output/now_8x/train/ours_15000/gt/00023.png', 
    output_name='error_progressive_vs_gt.png'
)
