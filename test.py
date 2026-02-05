import torch
import argparse
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import os
import time
import re

from utils.test_dataset import dehaze_test_dataset
from models.model_convnext import fusion_net
from cal_parameters import count_parameters

parser = argparse.ArgumentParser(description='Shadow')
parser.add_argument('--test_dir', type=str, default='C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/dataset/ntire26_dataset/')  #./ShadowDataset/ntire25_sh_rem_valid_inp/
parser.add_argument('--output_dir', type=str, default='C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/dataset/results/')
parser.add_argument('-test_batch_size', help='Set the testing batch size', default=1, type=int)
parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., cuda:0, cpu)')
args = parser.parse_args()

if not os.path.exists(args.output_dir + '/'):
    os.makedirs(args.output_dir + '/', exist_ok=True)

test_dataset = dehaze_test_dataset(args.test_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=0)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = fusion_net()


# UPDATE THIS LINE to match your actual filename!
# If your model is in a "models" folder, use: 'models/your_model_name.pth'
model_path = 'C:/Users/dax23/OneDrive/Documents/Projects/Fusion Shadow Removal/models/shadowremoval.pkl'

if os.path.exists(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=True)
    print(f'Loading removal_model success from: {model_path}')
else:
    raise FileNotFoundError(f"CRITICAL ERROR: Model file not found at: {model_path}")

model = model.to(device)

total_time = 0
with torch.no_grad():
    model.eval()

    start_time = time.time()
    for batch_idx, (input, name) in enumerate(test_loader):
        print(name[0])
        input = input.to(device)
        frame_out = model(input)
        frame_out = frame_out.to(device)

        name = re.findall("\d+", str(name))
        imwrite(frame_out, os.path.join(args.output_dir, str(name[0]) + '.png'), value_range=(0, 1))

    end_time = time.time()

elapsed_time = end_time - start_time
print(f"Runtime per image [s]: {elapsed_time / 75:.6f} s")
print(f"Params.(M): {count_parameters(model) / 1e6} M")  # 373192376
# Runtime per image [s]: 0.845790 s
# Params.(M): 373.192376 M