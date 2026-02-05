import os
import cv2
import numpy as np
from tqdm import tqdm
import argparse

# ================= CONFIGURATION =================
parser = argparse.ArgumentParser(description='Generate Shadow Masks')
parser.add_argument('--input_dir', type=str, default='C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/dataset/ntire24_shrem_valid_inp/LQ_padding', 
                    help='Path to input shadow images (e.g., train_A)')
parser.add_argument('--output_dir', type=str, default='C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/dataset/masks', 
                    help='Path to save generated masks (e.g., train_B)')
args = parser.parse_args()
# =================================================

def generate_masks():
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(files)} images. Generating masks...")

    for filename in tqdm(files):
        # 1. Read Image
        img_path = os.path.join(args.input_dir, filename)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Error reading {filename}")
            continue

        # 2. Convert to LAB Color Space
        # LAB separates Lightness (L) from Color (A, B). 
        # Shadows are mostly differences in Lightness.
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]

        # 3. Apply Otsu's Thresholding
        # Inverts the threshold so Dark (Shadow) = White (255), Light = Black (0)
        # We use THRESH_BINARY_INV because shadows are darker than the threshold
        val, mask = cv2.threshold(l_channel, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # 4. Post-Processing (Morphology) to remove noise
        # Kernel size 3x3 or 5x5 usually works best
        kernel = np.ones((3,3), np.uint8)
        
        # Opening removes white noise (small white dots in background)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Closing fills small black holes inside the shadow
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # 5. Save
        save_path = os.path.join(args.output_dir, filename)
        cv2.imwrite(save_path, mask)

    print(f"\nDone! Masks saved to: {args.output_dir}")

if __name__ == '__main__':
    generate_masks()