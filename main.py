import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from edl_reconstruction import EvidentialUNet, CelebAColorizationDataset, visualize_results_interactive, train_model

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