import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= CONFIGURATION =================
INPUT_DIR = 'C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/dataset/ntire24_shrem_valid_inp/LQ_padding/'
# CORRECTION: Make sure REAL_MASK_DIR points to the Black/White masks you generated, NOT the results.
# Based on your previous message, you likely saved masks to:
REAL_MASK_DIR = 'C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/dataset/masks/'

RESULT_DIR = 'C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/dataset/TTA_results/'
OUTPUT_DIR = 'C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/dataset/results_refined/'
# =================================================

def find_file_by_name(directory, name_no_ext):
    """Searches for a file with the given name (ignoring extension) in the directory."""
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        path = os.path.join(directory, name_no_ext + ext)
        if os.path.exists(path):
            return path
    return None

def refine_shadows():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Get list of result files
    files = [f for f in os.listdir(RESULT_DIR) if f.endswith(('.png', '.jpg'))]
    print(f"Refining {len(files)} images...")

    success_count = 0

    for filename in tqdm(files):
        # 1. Parse filename (remove extension)
        name_no_ext = os.path.splitext(filename)[0]

        # 2. Find matching Input and Mask files (handling .jpg vs .png)
        path_input = find_file_by_name(INPUT_DIR, name_no_ext)
        path_mask = find_file_by_name(REAL_MASK_DIR, name_no_ext)
        path_result = os.path.join(RESULT_DIR, filename)

        # 3. Validation
        if path_input is None:
            # Try debugging common issue: maybe input has no leading zeros? 
            # e.g. Result is "0098", Input is "98"
            path_input = find_file_by_name(INPUT_DIR, str(int(name_no_ext)))
            
        if path_input is None:
            print(f"\n[ERROR] Could not find INPUT image for: {name_no_ext} in {INPUT_DIR}")
            continue
        if path_mask is None:
            print(f"\n[ERROR] Could not find MASK image for: {name_no_ext} in {REAL_MASK_DIR}")
            continue

        # 4. Load images
        img_input = cv2.imread(path_input)
        img_result = cv2.imread(path_result)
        img_mask = cv2.imread(path_mask, cv2.IMREAD_GRAYSCALE)

        if img_input is None or img_result is None or img_mask is None:
            print(f"Skipping {filename}: Read error.")
            continue

        # 5. Resize to match result
        h, w = img_result.shape[:2]
        if img_input.shape[:2] != (h, w):
            img_input = cv2.resize(img_input, (w, h))
        if img_mask.shape[:2] != (h, w):
            img_mask = cv2.resize(img_mask, (w, h))

        # 6. Soft Blending
        kernel = np.ones((5, 5), np.uint8)
        mask_dilated = cv2.dilate(img_mask, kernel, iterations=2)
        mask_blur = cv2.GaussianBlur(mask_dilated, (9, 9), 0)
        
        alpha = mask_blur.astype(float) / 255.0
        alpha = np.expand_dims(alpha, axis=2)

        refined = (img_result * alpha + img_input * (1.0 - alpha))

        cv2.imwrite(os.path.join(OUTPUT_DIR, filename), refined)
        success_count += 1

    print(f"\nDone! Successfully refined {success_count} images.")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    refine_shadows()