import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image as imwrite
import os
import time
import argparse
from tqdm import tqdm

from utils.train_dataset import dehaze_train_dataset, dehaze_valid_dataset
from models.model import final_net
from models.model_convnext import fusion_net
from models.Restormer.restormer_arch import Restormer
from models.archs.NAFNet_arch import NAFNet


# 参数设置
parser = argparse.ArgumentParser(description='Train Shadow Removal Model')
parser.add_argument('--train_dir', type=str, default='./ShadowDataset/ntire2025_sh_rem_train/', help='Path to training dataset')
parser.add_argument('--val_dir', type=str, default='./ShadowDataset/', help='Path to validation dataset')
parser.add_argument('--output_dir', type=str, default='./checkpoints_with_23datas_try4/', help='Path to save trained models and results')
parser.add_argument('--batch_size', type=int, default=4, help='Training batch size')  # 8
parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs') #200
parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate') #1e-4
parser.add_argument('--save_interval', type=int, default=10, help='Save pth interval')
parser.add_argument('--device', type=str, default='cuda:3', help='Device to use (e.g., cuda:0, cpu)')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_dataset = dehaze_train_dataset(args.train_dir)
#valid_dataset = dehaze_valid_dataset(args.val_dir)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)
#valid_loader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False, num_workers=20)

# 初始化模型
modelA = fusion_net().to(device)
#modelA = Restormer().to(device)
#modelA = final_net().to(device)
#modelA = NAFNet(img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]).to(device)

# 加载预训练权重（如果有）
try:
    modelA.load_state_dict(torch.load(os.path.join('weights', 'shadowremoval.pkl'), map_location='cpu'), strict=True)
    print('Loaded removal_model weights successfully')
except:
    print('Failed to load removal_model weights')
'''
try:
    modelA.enhancement_model.load_state_dict(torch.load(os.path.join('weights', 'refinement.pkl'), map_location='cpu'), strict=True)
    print('Loaded enhancement_model weights successfully')
except:
    print('Failed to load enhancement_model weights')
'''

l1_criterion = nn.L1Loss()
from losses.my_pytorch_msssim import MSSSIM
ms_ssim_criterion = MSSSIM()
from saicinpainting.training.losses.style_loss import PerceptualLoss
lpips_criterion = PerceptualLoss().to(device)
from losses.focal_frequency_loss import FocalFrequencyLoss as FFL
fft_criterion = FFL(loss_weight=1.0, alpha=1.0, log_matrix=True)  # initialize fft loss

optimizerA = optim.Adam(modelA.parameters(), lr=args.lr, weight_decay=1e-8)
#optimizerB = optim.Adam(modelB.parameters(), lr=args.lr)
#optimizer = optim.Adam(modelA.parameters(), lr=args.lr, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizerA, args.epochs, eta_min=5 * 1e-6)
#scheduler.step()

# 训练ModelA
print("Training Model A...")
for epoch in range(args.epochs):
    modelA.train()
    total_loss = 0.0
    val_loss_total = 0.0
    for shadows, gts in tqdm(train_loader):
        shadows, gts = shadows.to(device), gts.to(device)
        outputsA = modelA(shadows)

        l1_loss = l1_criterion(outputsA, gts)
        ms_ssim_loss = ms_ssim_criterion(outputsA, gts)
        lpips_loss = lpips_criterion(outputsA, gts)
        fft_loss = fft_criterion(outputsA, gts)
        #loss_total = l1_loss + 0.01 * lpips_loss + 0.2 * (1 - ms_ssim_loss)   # try3 Chen epoch65 best
        #loss_total = l1_loss + 0.02 * lpips_loss + 0.1 * fft_loss  #try4
        loss_total = l1_loss + 0.05 * lpips_loss + 0.1 * fft_loss + 0.2 * (1. - ms_ssim_loss) #try3  epoch65 best e100下降
        # val最佳 27.058(4) 0.8363(7) 0.0985(4)  loss_total = l1_loss + 0.01 * lpips_loss + 0.05 * fft_loss  # try1 epoch35 and 100 best
        # loss_total = l1_loss + 0.05 * lpips_loss + 0.1 * fft_loss   try2 best
        optimizerA.zero_grad()
        loss_total.backward()
        optimizerA.step()
        total_loss += loss_total.item()
    avg_loss = total_loss / len(train_loader)

    scheduler.step()

    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")

    if (epoch+1) % args.save_interval == 0:
        torch.save(modelA.state_dict(), f"{args.output_dir}/modelA_epoch_{epoch+1}.pth")
    '''
    # Validation Step
    modelA.eval()
    with torch.no_grad():
        for shadows, gts in tqdm(valid_loader):
            shadows, gts = shadows.to(device), gts.to(device)
            outputsA = modelA(shadows)
            loss1 = criterion(outputsA, gts)
            # loss2 = criterion2(outputsA, gts)
            loss3 = criterion3(outputsA, gts)
            # loss1 = L1_msssim_criterion(outputsA, gts)
            val_loss = loss1 + 0.01 * loss3  # + 0.2 * (1 - loss2)

            val_loss_total += val_loss.item()
        avg_val_loss = val_loss_total / len(valid_loader)

        print(f"Epoch [{epoch+1}/{args.epochs}], Val Loss: {avg_val_loss:.4f}")
    '''

torch.save(modelA.state_dict(), f"{args.output_dir}/modelA_final.pth")


'''
# 训练ModelB（使用ModelA的输出作为输入）
print("Training Model B...")
modelA.eval()  # 固定ModelA

for epoch in range(args.epochs):
    modelB.train()
    total_loss = 0.0
    for shadows, gts in train_loader:
        shadows, gts = shadows.to(device), gts.to(device)
        with torch.no_grad():
            outputsA = modelA(shadows)
        outputsB = (modelB(outputsA) * 0.05 + outputsA) / (1 + 0.05)
        #outputsB = modelB(outputsA)
        loss = criterion(outputsB, gts)
        optimizerB.zero_grad()
        loss.backward()
        optimizerB.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}")
    if (epoch+1) % args.save_interval == 0:
        torch.save(modelB.state_dict(), f"checkpoints/modelB_epoch_{epoch+1}.pth")
torch.save(modelB.state_dict(), "checkpoints/modelB_final.pth")
'''