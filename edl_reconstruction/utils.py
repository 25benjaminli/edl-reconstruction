import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageFilter
import os
import glob
from skimage.color import rgb2lab, lab2rgb
from matplotlib.widgets import Slider, CheckButtons
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .edl import NormalInvGammaConv2d
from .edl import evidential_regression


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
    

def visualize_results_interactive(model, dataset):
    device = next(model.parameters()).device
    model.eval()

    # Create figure with 5 subplots in a row
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    plt.subplots_adjust(bottom=0.25)

    ax_slider = plt.axes((0.2, 0.08, 0.6, 0.03))
    slider = Slider(ax_slider, 'Image Index', 0, len(dataset) - 1,
                    valinit=0, valstep=1, color='lightblue')

    # OOD splotch checkbox
    ax_check = plt.axes((0.42, 0.02, 0.16, 0.05))
    check = CheckButtons(ax_check, ['Add OOD splotch'], [False])
    ood_enabled = [False]

    def on_check(label):
        ood_enabled[0] = not ood_enabled[0]
        update(slider.val)

    check.on_clicked(on_check)

    def add_ood_splotch(L_tensor):
        out = L_tensor.clone()
        _, H, W = out.shape
        rng = np.random.default_rng()
        cx = rng.integers(W // 4, 3 * W // 4)
        cy = rng.integers(H // 4, 3 * H // 4)
        r  = rng.integers(max(1, W // 20), W // 10)
        ys, xs = np.ogrid[:H, :W]
        mask = torch.from_numpy(((xs - cx) ** 2 + (ys - cy) ** 2 <= r ** 2).astype(np.float32))
        out[0] = out[0] * (1 - mask) + 1.0 * mask
        return out

    img_displays = []
    img_displays.append(axes[0].imshow(
        np.zeros((128, 128)), cmap='gray', vmin=0, vmax=1))
    axes[0].axis('off')
    img_displays.append(axes[1].imshow(np.zeros((128, 128, 3))))
    axes[1].axis('off')
    img_displays.append(axes[2].imshow(np.zeros((128, 128, 3))))
    axes[2].axis('off')
    img_displays.append(axes[3].imshow(np.zeros((128, 128)), cmap='hot'))
    axes[3].axis('off')
    img_displays.append(axes[4].imshow(np.zeros((128, 128)), cmap='hot'))
    axes[4].axis('off')
    metrics_text = fig.text(0.5, 0.95, '', ha='center', va='top',
                            fontsize=12, weight='bold')

    def update(idx):
        idx = int(idx)
        with torch.no_grad():
            L, ab_gt = dataset[idx]
            if ood_enabled[0]:
                L = add_ood_splotch(L)

            mu, v, alpha, beta = model(L.unsqueeze(0).to(device))

            epistemic_np = (beta / (alpha - 1))[0].mean(0).cpu().numpy()
            aleatoric_np = (beta / (v * (alpha - 1)))[0].mean(0).cpu().numpy()

            L_display = (L[0].cpu().numpy() + 1.) * 50.
            ab_pred_np = mu[0].cpu().numpy() * 110.
            ab_gt_np   = ab_gt.cpu().numpy() * 110.

            rgb_pred = np.clip(lab2rgb(np.stack([L_display, ab_pred_np[0], ab_pred_np[1]], axis=2)), 0, 1)
            rgb_gt = np.clip(lab2rgb(np.stack([L_display, ab_gt_np[0],ab_gt_np[1]], axis=2)), 0, 1)

            ssim = calculate_ssim(rgb_gt, rgb_pred)
            psnr = calculate_psnr(rgb_gt, rgb_pred)

            img_displays[0].set_data(L_display / 100.)
            img_displays[0].set_clim(0, 1)
            img_displays[1].set_data(rgb_gt)
            img_displays[2].set_data(rgb_pred)
            img_displays[3].set_data(epistemic_np)
            img_displays[3].set_clim(epistemic_np.min(), epistemic_np.max())
            img_displays[4].set_data(aleatoric_np)
            img_displays[4].set_clim(aleatoric_np.min(), aleatoric_np.max())

            if not hasattr(update, 'epistemic_cbar'):
                update.epistemic_cbar = plt.colorbar(img_displays[3], ax=axes[3], fraction=0.046, pad=0.04)
                update.aleatoric_cbar = plt.colorbar(img_displays[4], ax=axes[4], fraction=0.046, pad=0.04)
            else:
                update.epistemic_cbar.update_normal(img_displays[3])
                update.aleatoric_cbar.update_normal(img_displays[4])

            ood_tag = ' + OOD' if ood_enabled[0] else ''
            axes[0].set_title(f'Grayscale Input (L){ood_tag}', fontsize=10)
            axes[1].set_title('Ground Truth', fontsize=10)
            axes[2].set_title('Colorized (μ)', fontsize=10)
            axes[3].set_title('Epistemic Uncertainty', fontsize=10)
            axes[4].set_title('Aleatoric Uncertainty', fontsize=10)
            metrics_text.set_text(f'Image {idx+1}/{len(dataset)} | SSIM: {ssim:.4f} | PSNR: {psnr:.2f} dB')
            fig.canvas.draw_idle()

    slider.on_changed(update)
    update(0)
    plt.show()

def train_model(model, train_loader, num_epochs=20, lamb=0.01):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print(f"Training on {device}")
    losses = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (corrupted_img, target) in enumerate(train_loader):
            corrupted_img, target = corrupted_img.to(device), target.to(device)

            optimizer.zero_grad()
            mu, v, alpha, beta = model(corrupted_img)

            # the loss should be the sum of losses per channel instead of flattening
            channel_losses = [
                evidential_regression(
                    (
                        mu[:, c].reshape(-1, 1),
                        v[:, c].reshape(-1, 1),
                        alpha[:, c].reshape(-1, 1),
                        beta[:, c].reshape(-1, 1),
                    ),
                    target[:, c].reshape(-1, 1),
                    lamb=lamb
                )
                for c in range(mu.shape[1])
            ]
            loss = torch.stack(channel_losses).sum()
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 50 == 0:
                print(
                    f"  Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.show()
    return model
