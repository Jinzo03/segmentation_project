import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from dataset_instance import InstanceSegmentationDataset, collate_fn

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4       # Detection models are memory-heavy; keep batch size modest
NUM_EPOCHS = 10      # Fast run to see it converge
LEARNING_RATE = 0.005
NUM_CLASSES = 2      # 0: Background, 1: Circle, 2: Square (Strictly defined this way)
DATA_DIR = "data_cells"

def get_instance_model(num_classes):
    """
    Loads a ResNet-50 Mask R-CNN model pre-trained on COCO, 
    and replaces its output heads to match our custom class count.
    """
    # Load the anchor backbone architecture
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # 1. Modify the Box Predictor Head (For Bounding Boxes)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 2. Modify the Mask Predictor Head (For Pixel Masks)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def train_one_epoch(loader, model, optimizer, scaler):
    model.train()
    loop = tqdm(loader, desc="Training Instance Model")
    epoch_loss = 0.0

    for images, targets in loop:
        # Move images and target dictionaries to the target device (GPU/CPU)
        images = list(img.to(DEVICE) for img in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)

        # Forward pass under mixed precision
        with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            # Mask R-CNN outputs a loss dictionary directly during training mode
            loss_dict = model(images, targets)
            
            # Sum up all internal losses: box regression, classification, and mask segmentation
            losses = sum(loss for loss in loss_dict.values())

        # Backward pass
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += losses.item()
        loop.set_postfix(loss=losses.item())

    return epoch_loss / len(loader)

def main():
    print(f"Using device: {DEVICE}")

    # Augmented transformations including a horizontal flip to clear the bbox warning
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    # Initialize Dataset and Loader using custom collate configuration
    dataset = InstanceSegmentationDataset(root_dir=DATA_DIR, transform=train_transform)
    loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collate_fn,
        pin_memory=(DEVICE == "cuda")
    )

    # Build the modified model architecture
    model = get_instance_model(num_classes=NUM_CLASSES).to(DEVICE)
    
    # Setup optimizer and gradient scaler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    # Training Loop
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        avg_loss = train_one_epoch(loader, model, optimizer, scaler)
        print(f"Epoch Loss: {avg_loss:.4f}")
        
        # Save checkpoints
        torch.save({"state_dict": model.state_dict()}, f"instance_checkpoint_{epoch+1}.pth.tar")

    print("Training complete! Model checkpoints saved.")

if __name__ == "__main__":
    main()