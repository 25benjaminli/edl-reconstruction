import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
import os
import glob
from skimage.color import rgb2lab, lab2rgb


def calculate_ssim(img1, img2):
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    img1, img2 = img1.ravel(), img2.ravel()
    C1, C2 = (0.01) ** 2, (0.03) ** 2
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    return float(((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) /
                 ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)))


def calculate_psnr(img1, img2):
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    mse = np.mean((img1 - img2) ** 2)
    return float('inf') if mse == 0 else float(20 * np.log10(1.0 / np.sqrt(mse)))


class CelebAColorizationDataset(Dataset):
    """Gives L in [-1,1], ab in [-1,1]."""

    def __init__(self, root='./data/celeba/img_align_celeba', img_size=64, max_images=None):
        self.image_paths = sorted(glob.glob(os.path.join(root, '*.jpg')))
        if not self.image_paths:
            raise RuntimeError(f"No images found in {root}.")
        if max_images is not None:
            self.image_paths = self.image_paths[:max_images]
        print(f"Loaded {len(self.image_paths)} images from {root}")
        self.transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(img_size),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_pil = self.transform(Image.open(self.image_paths[idx]).convert('RGB'))
        img_lab = transforms.ToTensor()(rgb2lab(np.array(img_pil)).astype("float32"))
        L  = img_lab[[0], ...] / 50. - 1.
        ab = img_lab[[1, 2], ...] / 110.
        return L, ab