import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import os
import time
import re

# Import your modules
from utils.test_dataset import dehaze_test_dataset
from models.model_convnext import fusion_net
from cal_parameters import count_parameters

# ================= CONFIGURATION =================
parser = argparse.ArgumentParser(description='Shadow')
parser.add_argument('--test_dir', type=str, default='C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/dataset/ntire26_dataset/') 
parser.add_argument('--output_dir', type=str, default='C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/dataset/TTA_results/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0, cpu)')
args = parser.parse_args()
# =================================================

# Create output folder if missing
if not os.path.exists(args.output_dir + '/'):
    os.makedirs(args.output_dir + '/', exist_ok=True)

# Dataset setup (Keeping num_workers=0 for Windows stability)
test_dataset = dehaze_test_dataset(args.test_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize Model
model = fusion_net()

# --- MODEL LOADING (Fixed to crash if file missing) ---
# UPDATE THIS LINE WITH YOUR EXACT FILENAME
model_path = 'C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/models/shadowremoval.pkl' 

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    try:
        model.load_state_dict(checkpoint, strict=True)
    except RuntimeError as e:
        # Sometimes checkpoints are saved as 'state_dict' or inside a key
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            print(f"Warning: Strict loading failed, trying non-strict. Error: {e}")
            model.load_state_dict(checkpoint, strict=False)
            
    print(f'Loading removal_model success from: {model_path}')
else:
    raise FileNotFoundError(f"CRITICAL ERROR: Model file not found at: {model_path}")

model = model.to(device)
model.eval()

# ================= TTA FUNCTION =================
def tta_forward(model, x):
    """
    Smart TTA (2x):
    Only Horizontal Flip. 
    Vertical flips confuse shadow models because light usually comes from above.
    """
    outputs = []
    
    # 1. Original
    out = model(x)
    outputs.append(out)
    
    # 2. Horizontal Flip (Safe for shadows)
    # dim 3 is width
    x_lr = torch.flip(x, [3]) 
    out_lr = model(x_lr)
    outputs.append(torch.flip(out_lr, [3])) # Flip back
    
    # Average outputs
    final_out = torch.stack(outputs).mean(dim=0)
    return final_out

total_time = 0
print("Starting Inference with TTA...")

with torch.no_grad():
    start_time = time.time()
    
    for batch_idx, (input, name) in enumerate(test_loader):
        print(f"Processing: {name[0]}")
        input = input.to(device)
        
        # USE TTA FUNCTION INSTEAD OF DIRECT CALL
        frame_out = tta_forward(model, input)
        
        frame_out = frame_out.to(device)

        # Handle filename parsing
        filename_only = re.findall("\d+", str(name))
        save_name = str(filename_only[0]) if filename_only else str(name[0]).split('.')[0]
        
        # Save image (using value_range for newer torchvision)
        imwrite(frame_out, os.path.join(args.output_dir, save_name + '.png'), value_range=(0, 1))

    end_time = time.time()

elapsed_time = end_time - start_time
print(f"Total Runtime: {elapsed_time:.2f} s")
print(f"Runtime per image: {elapsed_time / len(test_loader):.4f} s")
print(f"Params.(M): {count_parameters(model) / 1e6:.2f} M")