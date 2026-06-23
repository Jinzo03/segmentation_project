import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 15
LR = 3e-4
DATA_DIR = "synthetic_dataset/images"
PATCH_SIZE = 8
EMBED_DIM = 128
IMAGE_SIZE = 64

# --- 1. Dataset with Pixel-Perfect Masks ---
class SegmentationDataset(Dataset):
    def __init__(self, data_dir, image_size=64):
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_paths = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".png")
        ])

        if len(self.image_paths) == 0:
            raise ValueError(f"No .png images found in: {data_dir}")

        rng = np.random.default_rng(42)
        self.labels = rng.integers(0, 2, size=len(self.image_paths))
        self.anomaly_coords = [
            (int(rng.integers(20, 44)), int(rng.integers(20, 44)))
            for _ in range(len(self.image_paths))
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read image: {self.image_paths[idx]}")

        if img.shape[:2] != (self.image_size, self.image_size):
            img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)

        label = int(self.labels[idx])
        mask = np.zeros_like(img, dtype=np.uint8)

        if label == 1:
            ix, iy = self.anomaly_coords[idx]

            # Draw on the image
            cv2.circle(img, (ix, iy), 6, 255, -1)
            cv2.circle(img, (ix + 5, iy - 4), 3, 200, -1)

            # Draw the exact same shapes on the mask
            cv2.circle(mask, (ix, iy), 6, 255, -1)
            cv2.circle(mask, (ix + 5, iy - 4), 3, 255, -1)

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        mask_tensor = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0

        return img_tensor, mask_tensor

# --- 2. The TransUNet Architecture ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleTransUNet(nn.Module):
    def __init__(self, image_size=64, patch_size=8, in_channels=1, embed_dim=128):
        super().__init__()
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size ** 2

        # High-res skip branch
        self.skip_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Patch embedding + transformer
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads=4) for _ in range(3)])
        self.norm = nn.LayerNorm(embed_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        )

        # Final fusion
        self.final_fusion = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        B = x.shape[0]

        skip_features = self.skip_extractor(x)  # [B, 32, 64, 64]

        vit_x = self.patch_embed(x).flatten(2).transpose(1, 2)  # [B, N, C]
        vit_x = vit_x + self.pos_embedding
        vit_x = self.blocks(vit_x)
        vit_x = self.norm(vit_x)

        vit_x = vit_x.transpose(1, 2).reshape(B, -1, self.grid_size, self.grid_size)
        decoded_x = self.decoder(vit_x)  # [B, 32, 64, 64]

        fused_x = torch.cat([decoded_x, skip_features], dim=1)  # [B, 64, 64, 64]
        return self.final_fusion(fused_x)

# --- 3. Metrics ---
def calculate_segmentation_metrics(pred_masks, true_masks, threshold=0.5):
    pred_binary = (pred_masks > threshold).float()
    true_binary = true_masks.float()

    pred_flat = pred_binary.reshape(-1)
    true_flat = true_binary.reshape(-1)

    intersection = (pred_flat * true_flat).sum()
    total_pixels_combined = pred_flat.sum() + true_flat.sum()
    union = total_pixels_combined - intersection

    smooth = 1e-6
    iou = (intersection + smooth) / (union + smooth)
    dice = (2.0 * intersection + smooth) / (total_pixels_combined + smooth)

    return iou.item(), dice.item()

# --- 4. Sim-to-Real Stress Test ---
def stress_test_model(model):
    print("\nInitiating Out-of-Distribution Stress Test...")

    img = np.ones((64, 64), dtype=np.uint8) * 60
    cv2.ellipse(img, (32, 32), (24, 16), 45, 0, 360, 180, -1)

    noise = np.random.normal(0, 40, (64, 64)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.circle(img, (22, 42), 6, 250, -1)

    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0

    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
        pred_mask = pred[0].cpu().squeeze().numpy()

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Messy 'Real-World' Cell")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask, cmap="magma")
    plt.title("AI Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("stress_test_result.png")
    plt.close()
    print("Stress test complete. Check 'stress_test_result.png'.")

# --- 5. Training Engine ---
def train_segmenter():
    print(f"Booting up Segmentation Engine on {DEVICE}...")

    dataset = SegmentationDataset(DATA_DIR, image_size=IMAGE_SIZE)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise ValueError("Dataset is too small to split into train/val sets.")

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleTransUNet(patch_size=PATCH_SIZE, embed_dim=EMBED_DIM).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    history_loss = []
    history_val_loss = []
    history_iou = []
    history_dice = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history_loss.append(avg_loss)

        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        total_dice = 0.0

        sample_imgs = None
        sample_masks = None
        sample_preds = None

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                raw_preds = model(imgs)
                loss = criterion(raw_preds, masks)
                val_loss += loss.item()

                prob_preds = torch.sigmoid(raw_preds)
                batch_iou, batch_dice = calculate_segmentation_metrics(prob_preds, masks)
                total_iou += batch_iou
                total_dice += batch_dice

                if sample_imgs is None:
                    sample_imgs = imgs.detach().cpu()
                    sample_masks = masks.detach().cpu()
                    sample_preds = prob_preds.detach().cpu()

        avg_val_loss = val_loss / len(val_loader)
        avg_iou = total_iou / len(val_loader)
        avg_dice = total_dice / len(val_loader)

        history_val_loss.append(avg_val_loss)
        history_iou.append(avg_iou)
        history_dice.append(avg_dice)

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"IoU: {avg_iou:.4f} | "
            f"Dice: {avg_dice:.4f}"
        )

    # Visualize the first validation sample
    img_viz = sample_imgs[0].squeeze().numpy()
    true_mask_viz = sample_masks[0].squeeze().numpy()
    pred_mask_viz = sample_preds[0].squeeze().numpy()

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 4, 1)
    plt.plot(history_loss, label="Train Loss")
    plt.plot(history_val_loss, label="Val Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 4, 2)
    plt.imshow(img_viz, cmap="gray")
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(true_mask_viz, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(pred_mask_viz, cmap="magma")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("segmentation_results.png")
    plt.close()

    print("\nTraining complete! Look at 'segmentation_results.png' to see the pixel mapping.")
    stress_test_model(model)

if __name__ == "__main__":
    train_segmenter()