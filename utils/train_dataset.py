import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


# =========================
# Helper
# =========================
def load_img(path):
    img = Image.open(path).convert('RGB')
    return TF.to_tensor(img)  # [0,1]


# =========================
# TRAIN DATASET
# =========================
class dehaze_train_dataset(Dataset):
    def __init__(self, train_dir, patch_size=512):
        super().__init__()
        self.patch_size = patch_size
        self.epoch = 0  # for curriculum augmentation

        self.root_input = os.path.join(train_dir, 'LQ/')
        self.root_target = os.path.join(train_dir, 'HQ/')

        self.list_train = sorted(os.listdir(self.root_input))
        self.file_len = len(self.list_train)

    def __len__(self):
        return self.file_len

    # -------------------------
    # Random crop (paired)
    # -------------------------
    def random_crop(self, inp, tgt):
        _, h, w = inp.shape
        ps = self.patch_size

        if h < ps or w < ps:
            return inp, tgt

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)

        inp = inp[:, top:top+ps, left:left+ps]
        tgt = tgt[:, top:top+ps, left:left+ps]

        return inp, tgt

    # -------------------------
    # Geometric augmentation
    # -------------------------
    def augment(self, inp, tgt):
        # Horizontal flip
        if random.random() < 0.5:
            inp = TF.hflip(inp)
            tgt = TF.hflip(tgt)

        # Vertical flip
        if random.random() < 0.5:
            inp = TF.vflip(inp)
            tgt = TF.vflip(tgt)

        # Rotation (safe angles only)
        if random.random() < 0.5:
            angle = random.choice([0, 90, 180, 270])
            inp = TF.rotate(inp, angle)
            tgt = TF.rotate(tgt, angle)

        return inp, tgt

    # -------------------------
    # Input-only degradation
    # -------------------------
    def degrade_input(self, inp):
        # Gaussian noise
        if random.random() < 0.3:
            noise = torch.randn_like(inp) * random.uniform(0.003, 0.01)
            inp = torch.clamp(inp + noise, 0.0, 1.0)

        # Gamma augmentation (shadow realism)
        if random.random() < 0.3:
            gamma = random.uniform(0.8, 1.2)
            inp = torch.clamp(inp ** gamma, 0.0, 1.0)

        # Contrast compression (haze-like)
        if random.random() < 0.3:
            mean = inp.mean(dim=(1, 2), keepdim=True)
            inp = torch.clamp(mean + 0.8 * (inp - mean), 0.0, 1.0)

        return inp

    # -------------------------
    # __getitem__
    # -------------------------
    def __getitem__(self, index):
        name = self.list_train[index]

        inp = load_img(os.path.join(self.root_input, name))
        tgt = load_img(os.path.join(self.root_target, name))

        # Paired random crop
        inp, tgt = self.random_crop(inp, tgt)

        # Paired geometric augmentation
        inp, tgt = self.augment(inp, tgt)

        # Curriculum degradation (input only)
        if self.epoch > 30:
            inp = self.degrade_input(inp)

        return inp, tgt


# =========================
# VALID DATASET (NO AUG)
# =========================
class dehaze_valid_dataset(Dataset):
    def __init__(self, val_dir):
        super().__init__()

        self.root_input = os.path.join(val_dir, 'ntire23_sr_valid_inp/')
        self.root_target = os.path.join(val_dir, 'ntire23_sr_valid_gt/')

        self.list_val = sorted(os.listdir(self.root_input))
        self.file_len = len(self.list_val)

    def __len__(self):
        return self.file_len

    def __getitem__(self, index):
        name = self.list_val[index]

        inp = load_img(os.path.join(self.root_input, name))
        tgt = load_img(os.path.join(self.root_target, name))

        return inp, tgt