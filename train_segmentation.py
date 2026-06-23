import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 15
LR = 3e-4
DATA_DIR = "synthetic_dataset/images"
PATCH_SIZE = 8
EMBED_DIM = 128

# --- 1. Dataset with Pixel-Perfect Masks ---
class SegmentationDataset(Dataset):
    def __init__(self, data_dir):
        self.image_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png")])
        np.random.seed(42)
        # 50% chance of anomaly
        self.labels = np.random.choice([0, 1], size=len(self.image_paths))
        self.anomaly_coords = [(np.random.randint(20, 44), np.random.randint(20, 44)) for _ in range(len(self.image_paths))]
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        label = self.labels[idx]
        
        # Create a blank black mask
        mask = np.zeros_like(img)
        
        if label == 1:
            ix, iy = self.anomaly_coords[idx]
            # Draw on the Image
            cv2.circle(img, (ix, iy), 6, 255, -1)
            cv2.circle(img, (ix+5, iy-4), 3, 200, -1)
            # Draw the EXACT same shapes on the Mask
            cv2.circle(mask, (ix, iy), 6, 255, -1)
            cv2.circle(mask, (ix+5, iy-4), 3, 255, -1)
            
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        # Masks are binary: 0.0 for background, 1.0 for anomaly
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
        
        # --- THE SKIP CONNECTION (High-Res Memory) ---
        # We extract crisp 64x64 pixel features before the Transformer ruins the resolution
        self.skip_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # --- ENCODER (The Big Picture) ---
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads=4) for _ in range(3)])
        self.norm = nn.LayerNorm(embed_dim)
        
        # --- DECODER (Upscaling) ---
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 64, kernel_size=2, stride=2), # 8x8 -> 16x16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),        # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)         # 32x32 -> 64x64 (Matches skip!)
        )
        
        # --- FUSION LAYER ---
        # We concatenate the 32 channels from the decoder and 32 channels from the skip connection
        self.final_fusion = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1) # Squash to 1 channel mask prediction
        )

    def forward(self, x):
        B = x.shape[0]
        
        # 1. Save the high-res details
        skip_features = self.skip_extractor(x) # Shape: [B, 32, 64, 64]
        
        # 2. Process the low-res global context
        vit_x = self.patch_embed(x).flatten(2).transpose(1, 2)
        vit_x = vit_x + self.pos_embedding
        vit_x = self.blocks(vit_x)
        vit_x = self.norm(vit_x)
        
        # 3. Upscale the ViT features back to 64x64
        vit_x = vit_x.transpose(1, 2).reshape(B, -1, self.grid_size, self.grid_size)
        decoded_x = self.decoder(vit_x) # Shape: [B, 32, 64, 64]
        
        # 4. THE MAGIC: Concatenate the big picture with the fine details along the channel dimension
        fused_x = torch.cat([decoded_x, skip_features], dim=1) # Shape: [B, 64, 64, 64]
        
        # 5. Final pass to predict exactly where the pixels are
        return self.final_fusion(fused_x)

# --- 3. Training Engine ---
def train_segmenter():
    print(f"Booting up Segmentation Engine on {DEVICE}...")
    
    dataset = SegmentationDataset(DATA_DIR)
    train_size = 400
    val_size = len(dataset) - train_size
    
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SimpleTransUNet(patch_size=PATCH_SIZE, embed_dim=EMBED_DIM).to(DEVICE)
    
    # Loss function for predicting binary (0 or 1) pixels
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    history_loss = []
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(imgs) # Model outputs [B, 1, 64, 64] prediction
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        history_loss.append(avg_loss)
        print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_loss:.4f}")

    # --- Plot the Loss and Visualize a Prediction ---
    model.eval()
    with torch.no_grad():
        # Grab one batch from validation
        val_imgs, val_masks = next(iter(val_loader))
        val_imgs, val_masks = val_imgs.to(DEVICE), val_masks.to(DEVICE)
        
        # Get raw predictions and apply sigmoid to squash to 0.0 - 1.0 probability
        raw_preds = model(val_imgs)
        pred_masks = torch.sigmoid(raw_preds)

    # Let's visualize the first image in the validation batch
    img_viz = val_imgs[0].cpu().squeeze().numpy()
    true_mask_viz = val_masks[0].cpu().squeeze().numpy()
    pred_mask_viz = pred_masks[0].cpu().squeeze().numpy()

    plt.figure(figsize=(15, 4))
    
    plt.subplot(1, 4, 1)
    plt.plot(history_loss, color='red', label="BCE Loss")
    plt.title("Segmentation Loss Curve")
    plt.xlabel("Epochs")
    
    plt.subplot(1, 4, 2)
    plt.imshow(img_viz, cmap="gray")
    plt.title("Input Cell")
    plt.axis("off")
    
    plt.subplot(1, 4, 3)
    plt.imshow(true_mask_viz, cmap="gray")
    plt.title("Ground Truth Mask")
    plt.axis("off")
    
    plt.subplot(1, 4, 4)
    plt.imshow(pred_mask_viz, cmap="magma") # Heatmap style prediction
    plt.title("AI Predicted Mask")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("segmentation_results.png")
    print("\nTraining complete! Look at 'segmentation_results.png' to see the AI's pixel mapping.")
    stress_test_model(model)
# --- 4. The Sim-to-Real Stress Test ---
def stress_test_model(model):
    print("\nInitiating Out-of-Distribution Stress Test...")
    
    # 1. Create a "real" messy cell: gray background, irregular ellipse shape
    img = np.ones((64, 64), dtype=np.uint8) * 60
    cv2.ellipse(img, (32, 32), (24, 16), 45, 0, 360, 180, -1) 
    
    # 2. Add heavy real-world microscope noise
    noise = np.random.normal(0, 40, (64, 64)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 3. Add the infection anomaly in a totally random spot
    cv2.circle(img, (22, 42), 6, 250, -1)
    
    # 4. Run Inference
    img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE) / 255.0
    
    model.eval()
    with torch.no_grad():
        pred = torch.sigmoid(model(img_tensor))
        pred_mask = pred[0].cpu().squeeze().numpy()
        
    # 5. Visualize
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
    print("Stress test complete. Check 'stress_test_result.png'.")
if __name__ == "__main__":
    train_segmenter()