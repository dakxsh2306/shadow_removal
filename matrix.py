print("Script starting...", flush=True) # 1. Check if python is running at all

import os
import cv2
import numpy as np
import torch
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# ================= CONFIGURATION =================
# Path to your clean "Ground Truth" images
GT_PATH = 'C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/dataset/ntire24_shrem_valid_gt/'

# Path to the images your model just generated
GEN_PATH = 'C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/dataset/TTA_results/'

# Device setup
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# =================================================

def calculate_metrics():
    # 2. Check if paths exist
    if not os.path.exists(GT_PATH):
        print(f"ERROR: GT_PATH does not exist: {GT_PATH}")
        return
    if not os.path.exists(GEN_PATH):
        print(f"ERROR: GEN_PATH does not exist: {GEN_PATH}")
        return

    print(f"Initializing LPIPS on {device}...", flush=True)
    
    # 3. This is where it likely hangs (Downloading weights)
    try:
        loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        print("LPIPS Initialized successfully.", flush=True)
    except Exception as e:
        print(f"Failed to initialize LPIPS: {e}")
        return

    gt_files = sorted(os.listdir(GT_PATH))
    print(f"Found {len(gt_files)} files in GT folder.", flush=True)
    gen_files = sorted(os.listdir(GEN_PATH))
    
    # Filter only images (png, jpg)
    gt_files = [f for f in gt_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    psnr_scores = []
    ssim_scores = []
    lpips_scores = []

    print(f"Found {len(gt_files)} Ground Truth images.")
    print(f"Processing...")

    for filename in gt_files:
        # Construct full paths
        path_gt = os.path.join(GT_PATH, filename)
        path_gen = os.path.join(GEN_PATH, filename)

        # Check if the generated file exists
        if not os.path.exists(path_gen):
            print(f"Warning: Missing generated file for {filename}, skipping.")
            continue

        # --- Load Images for PSNR/SSIM (OpenCV uses BGR, convert to RGB) ---
        # Read as [0, 255]
        img_gt = cv2.imread(path_gt)
        img_gen = cv2.imread(path_gen)

        if img_gt is None or img_gen is None:
            print(f"Error reading {filename}")
            continue

        # Ensure same size (Generated might be slightly different due to padding)
        if img_gt.shape != img_gen.shape:
            img_gen = cv2.resize(img_gen, (img_gt.shape[1], img_gt.shape[0]))

        # Convert to RGB
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        img_gen = cv2.cvtColor(img_gen, cv2.COLOR_BGR2RGB)

        # --- Calculate PSNR & SSIM (using 0-255 range or 0-1 range) ---
        # We prefer float 0-1 for standard calculations
        img_gt_norm = img_gt.astype(np.float32) / 255.0
        img_gen_norm = img_gen.astype(np.float32) / 255.0

        # PSNR
        psnr_val = psnr_metric(img_gt_norm, img_gen_norm, data_range=1.0)
        psnr_scores.append(psnr_val)

        # SSIM (multichannel/channel_axis depends on scikit-image version)
        try:
            # Newer scikit-image
            ssim_val = ssim_metric(img_gt_norm, img_gen_norm, data_range=1.0, channel_axis=2)
        except TypeError:
            # Older scikit-image
            ssim_val = ssim_metric(img_gt_norm, img_gen_norm, multichannel=True)
            
        ssim_scores.append(ssim_val)

        # --- Calculate LPIPS ---
        # LPIPS expects NCHW float tensor, normalized to [-1, 1]
        t_gt = torch.from_numpy(img_gt_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
        t_gen = torch.from_numpy(img_gen_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

        # Normalize to [-1, 1]
        t_gt = t_gt * 2 - 1
        t_gen = t_gen * 2 - 1

        with torch.no_grad():
            lpips_val = loss_fn_alex(t_gt, t_gen)
        
        lpips_scores.append(lpips_val.item())

        print(f"Processed {filename} | PSNR: {psnr_val:.2f} | SSIM: {ssim_val:.4f} | LPIPS: {lpips_val.item():.4f}")

    # --- Final Results ---
    print("\n" + "="*30)
    print("FINAL RESULTS")
    print("="*30)
    print(f"Images Processed: {len(psnr_scores)}")
    print(f"Average PSNR:  {np.mean(psnr_scores):.4f} dB")
    print(f"Average SSIM:  {np.mean(ssim_scores):.4f}")
    print(f"Average LPIPS: {np.mean(lpips_scores):.4f}")
    print("="*30)

if __name__ == '__main__':
    calculate_metrics()