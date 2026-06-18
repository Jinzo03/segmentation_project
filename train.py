import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader 
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from model import UNet
from dataset import SegmentationDataset

# Hyperparameters
LEARNING_RATE = 1e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 20
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
TRAIN_IMG_DIR = "data/train_images/"
TRAIN_MASK_DIR = "data/train_masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice_loss = 1 - ((2.0 * intersection + 1e-8) / (pred.sum() + target.sum() + 1e-8))
        return bce_loss + dice_loss

def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Training")
    epoch_loss = 0.0

    for data, targets in loop:
        data = data.to(DEVICE)
        targets = targets.float().unsqueeze(1).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)

def check_accuracy(loader, model):
    model.eval()
    dice_score = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.float().unsqueeze(1).to(DEVICE)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    model.train()
    return dice_score / len(loader)

def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # FIX 1: defined val_transform (was undefined "val_transforms")
    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    train_ds = SegmentationDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=(DEVICE=="cuda"), shuffle=True)

    # FIX 2: use val_transform instead of val_transforms
    val_ds = SegmentationDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=(DEVICE=="cuda"))

    # FIX 3: removed misplaced validation check block; training loop runs unconditionally
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        # FIX 4: capture train loss and print it
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print(f"Train Loss: {train_loss:.4f}")

        if len(val_loader) > 0:
            val_dice = check_accuracy(val_loader, model)
            print(f"Validation Dice score: {val_dice:.4f}")
        else:
            print("WARNING: Validation loader is empty! Skipping validation.")

        torch.save({"state_dict": model.state_dict()}, f"checkpoint_{epoch+1}.pth.tar")

if __name__ == "__main__":
    main()