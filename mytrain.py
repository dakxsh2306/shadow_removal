import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import copy

# =========================
# Dataset & Model Imports
# =========================
from utils.train_dataset import dehaze_train_dataset, dehaze_valid_dataset
from models.model_convnext import fusion_net

# =========================
# Argument Parser
# =========================
parser = argparse.ArgumentParser(description='Fine-tune Shadow Removal Model')

parser.add_argument('--train_dir', type=str, default='C:\Users\dax23\OneDrive\Documents\Projects\Fusion Shadow Removal\dataset\ntire26_dataset')
parser.add_argument('--val_dir', type=str, default='/path/to/your/val_dataset/')
parser.add_argument('--output_dir', type=str, default='./checkpoints_finetune/')

parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--epochs', type=int, default=80)
parser.add_argument('--lr', type=float, default=5e-6)

parser.add_argument('--save_interval', type=int, default=20)
parser.add_argument('--device', type=str, default='cuda:0')

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =========================
# Datasets & Loaders
# =========================
train_dataset = dehaze_train_dataset(args.train_dir)
val_dataset = dehaze_valid_dataset(args.val_dir)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# =========================
# Model
# =========================
model = fusion_net().to(device)

# =========================
# Load Pretrained Weights
# =========================
try:
    ckpt = torch.load('weights/shadowremoval.pkl', map_location='cpu')
    model.load_state_dict(ckpt, strict=False)
    print("Loaded pretrained weights (strict=False)")
except Exception as e:
    print("Failed to load pretrained weights:", e)

# =========================
# Freeze backbone (warmup)
# =========================
for name, param in model.named_parameters():
    if "head" not in name and "output" not in name:
        param.requires_grad = False

print("Backbone frozen (warmup phase)")

# =========================
# EMA SETUP (CRITICAL)
# =========================
ema_decay = 0.999
ema_model = copy.deepcopy(model).eval()
for p in ema_model.parameters():
    p.requires_grad = False

def update_ema(model, ema_model, decay):
    with torch.no_grad():
        msd = model.state_dict()
        for k, v in ema_model.state_dict().items():
            v.copy_(v * decay + msd[k] * (1. - decay))

# =========================
# Loss Functions
# =========================
l1_criterion = nn.L1Loss()

from losses.my_pytorch_msssim import MSSSIM
ms_ssim_criterion = MSSSIM()

from saicinpainting.training.losses.style_loss import PerceptualLoss
lpips_criterion = PerceptualLoss().to(device)

from losses.focal_frequency_loss import FocalFrequencyLoss
fft_criterion = FocalFrequencyLoss(loss_weight=1.0, alpha=1.0, log_matrix=True)

# =========================
# Optimizer & Scheduler
# =========================
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr,
    weight_decay=1e-8
)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.epochs,
    eta_min=1e-6
)

scaler = torch.cuda.amp.GradScaler()

# =========================
# Training Loop
# =========================
best_val_loss = float('inf')

print("🚀 Starting Fine-Tuning")

for epoch in range(args.epochs):
    model.train()
    train_dataset.epoch = epoch  # 🔥 activates curriculum augmentation
    train_loss = 0.0

    # -------------------------
    # Unfreeze backbone
    # -------------------------
    if epoch == 20:
        for p in model.parameters():
            p.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        print("🔓 Backbone unfrozen — full model training")

    for inp, gt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
        inp = inp.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(inp)

            if epoch < 20:
                loss = l1_criterion(out, gt) + 0.2 * (1.0 - ms_ssim_criterion(out, gt))
            else:
                loss = (
                    l1_criterion(out, gt)
                    + 0.05 * lpips_criterion(out, gt)
                    + 0.1 * fft_criterion(out, gt)
                    + 0.2 * (1.0 - ms_ssim_criterion(out, gt))
                )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        update_ema(model, ema_model, ema_decay)
        train_loss += loss.item()

    scheduler.step()
    avg_train_loss = train_loss / len(train_loader)

    # =========================
    # Validation (EMA MODEL!)
    # =========================
    ema_model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for inp, gt in val_loader:
            inp = inp.to(device)
            gt = gt.to(device)
            out = ema_model(inp)
            val_loss += l1_criterion(out, gt).item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"[Epoch {epoch+1}] Train: {avg_train_loss:.4f} | Val (EMA): {avg_val_loss:.4f}")

    # =========================
    # Save best EMA model
    # =========================
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(ema_model.state_dict(),
                   os.path.join(args.output_dir, "best_model_ema.pth"))
        print("✅ Best EMA model updated")

    if (epoch + 1) % args.save_interval == 0:
        torch.save(ema_model.state_dict(),
                   os.path.join(args.output_dir, f"epoch_{epoch+1}_ema.pth"))

# =========================
# Final Save
# =========================
torch.save(ema_model.state_dict(),
           os.path.join(args.output_dir, "final_model_ema.pth"))

print("🎉 Fine-tuning completed successfully")
