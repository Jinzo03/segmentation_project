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
LEARNING_RATE = 1e-4  # Adjusted slightly higher for multiclass convergence
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 20
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CLASSES = 3  # 0: Background, 1: Circle, 2: Square
TRAIN_IMG_DIR = "data/images/"  # Update path if using standard data split directories
TRAIN_MASK_DIR = "data/masks/"
VAL_IMG_DIR = "data/val_images/"
VAL_MASK_DIR = "data/val_masks/"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Training")
    epoch_loss = 0.0

    for data, targets in loop:
        data = data.to(DEVICE)
        
        # FIX 1: CrossEntropyLoss requires target to be LongTensor and shaped (N, H, W)
        # Removed .unsqueeze(1) and changed .float() to .long()
        targets = targets.long().to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            predictions = model(data)  # Output shape: (N, NUM_CLASSES, H, W)
            loss = loss_fn(predictions, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return epoch_loss / len(loader)

def check_mIoU(loader, model, num_classes=3):
    """
    Calculates Mean Intersection over Union (mIoU) across all classes.
    Replaces binary Dice score calculation for multi-class tasks.
    """
    model.eval()
    total_iou = 0.0
    total_batches = 0
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.long().to(DEVICE)  # Shape: (N, H, W)
            
            # FIX 2: Multiclass evaluation uses argmax over the channel dim (dim=1)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)  # Shape: (N, H, W)
            
            batch_iou = 0.0
            for cls in range(num_classes):
                pred_cls = (preds == cls)
                true_cls = (y == cls)
                
                intersection = (pred_cls & true_cls).sum().item()
                union = (pred_cls | true_cls).sum().item()
                
                if union == 0:
                    batch_iou += 1.0  # Avoid zero division if class isn't present in ground truth or pred
                else:
                    batch_iou += intersection / union
                    
            total_iou += (batch_iou / num_classes)
            total_batches += 1

    model.train()
    return total_iou / total_batches if total_batches > 0 else 0.0

def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # FIX 3: Configured U-Net model output channels to handle 3 unique prediction classes
    model = UNet(in_channels=3, out_channels=NUM_CLASSES).to(DEVICE)
    
    # Modern Optimization: Compile the network graph if using compatible system engines
    if hasattr(torch, 'compile') and DEVICE == 'cuda':
        model = torch.compile(model)
        
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    train_ds = SegmentationDataset(image_dir=TRAIN_IMG_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transform)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=(DEVICE=="cuda"), shuffle=True)

    val_ds = SegmentationDataset(image_dir=VAL_IMG_DIR, mask_dir=VAL_MASK_DIR, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=2, pin_memory=(DEVICE=="cuda"))

    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")

        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print(f"Train Loss: {train_loss:.4f}")

        if len(val_loader) > 0:
            # FIX 4: Hooked up the new multi-class metric
            val_miou = check_mIoU(val_loader, model, num_classes=NUM_CLASSES)
            print(f"Validation mIoU: {val_miou:.4f}")
        else:
            print("WARNING: Validation loader is empty! Skipping validation.")

        torch.save({"state_dict": model.state_dict()}, f"checkpoint_{epoch+1}.pth.tar")

if __name__ == "__main__":
    main()