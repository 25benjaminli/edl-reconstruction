import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import numpy as np
from modules.edl import evidential_regression
from PIL import Image
from skimage.color import rgb2lab, lab2rgb
import os
import glob
import argparse
from modules.model import EvidentialUNet
from modules.utils import CelebAColorizationDataset, calculate_ssim, calculate_psnr


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


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true',
                        help='Evaluate the model without training')
    args = parser.parse_args()

    all_dataset = CelebAColorizationDataset(
        root='./data/img_align_celeba',
        img_size=128,
        max_images=2500,
    )

    train_size = int(0.8 * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        all_dataset, [train_size, test_size]
    )

    model = EvidentialUNet(in_channels=1, out_channels=2)
    print(f"Train: {len(train_dataset)}  Test: {len(test_dataset)}")

    if not args.eval_only:
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
        model = train_model(model, train_loader, num_epochs=20, lamb=0.01)
        torch.save(model.state_dict(), 'evidential_celeba_model.pth')
    else:
        model.load_state_dict(torch.load("evidential_celeba_model.pth"))

    visualize_results_interactive(model, test_dataset)