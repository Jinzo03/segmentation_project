import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import math

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
EPOCHS = 15
LR = 3e-4          # Classical ViT learning rate
DATA_DIR = "synthetic_dataset/images"
PATCH_SIZE = 8
EMBED_DIM = 128    # Increased dimension to give the attention heads more capacity

# --- 1. Deterministic Dataset Loader ---
class ViTCellDataset(Dataset):
    def __init__(self, data_dir):
        self.image_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".png")])
        np.random.seed(42)
        self.labels = np.random.choice([0, 1], size=len(self.image_paths))
        self.anomaly_coords = [(np.random.randint(20, 44), np.random.randint(20, 44)) for _ in range(len(self.image_paths))]
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        label = self.labels[idx]
        
        if label == 1:
            ix, iy = self.anomaly_coords[idx]
            cv2.circle(img, (ix, iy), 6, 255, -1)
            cv2.circle(img, (ix+5, iy-4), 3, 200, -1)
            
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0
        return img_tensor, torch.tensor(label, dtype=torch.long)

# --- 2. The Transformer Engine ---
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        qkv = self.qkv_proj(x).chunk(3, dim=-1)
        Q, K, V = qkv
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        return self.out_proj(context)

# --- Replace your TransformerBlock and SimpleViT with these upgraded versions ---

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4):  # Added 4 independent attention heads
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # FIX: Using PyTorch's production-grade MultiheadAttention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=num_heads, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        # PyTorch MHA returns (output_tensor, attention_weights). We grab just the tensor [0]
        norm_x = self.norm1(x)
        attn_out, _ = self.attention(norm_x, norm_x, norm_x) # Query, Key, Value are all norm_x
        x = x + attn_out
        
        x = x + self.mlp(self.norm2(x))
        return x

class SimpleViT(nn.Module):
    def __init__(self, image_size=64, patch_size=8, in_channels=1, embed_dim=128, num_classes=2):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        
        # Stack 3 Transformer Blocks with Multi-Head Attention enabled
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, num_heads=4) for _ in range(3)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = x + self.pos_embedding
        x = self.blocks(x)
        x = self.norm(x)
        
        x, _ = torch.max(x, dim=1)
        return self.head(x)

# --- 3. UPGRADED: Production Vision Transformer with CLS Token ---
class SimpleViT(nn.Module):
    def __init__(self, image_size=64, patch_size=8, in_channels=1, embed_dim=128, num_classes=2):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # FIX 1: Introduce the standard Classification Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        
        # Position embeddings must now accommodate the patches PLUS the CLS token (num_patches + 1)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim) for _ in range(3)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        # Patchify and flatten: [B, 1, 64, 64] -> [B, 64, 128]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # Expand and prepend the CLS token to the front of the patch sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # Shape becomes [B, 65, 128]
        
        # Add the positional map
        x = x + self.pos_embedding
        
        # Pass through attention blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # FIX 2: Hybrid Max-Pooling routing instead of mean dilution
        # This isolates the highest token activation across the sequence
        x, _ = torch.max(x, dim=1)
        
        return self.head(x)

# --- 4. Running the Engine ---
def train_vit():
    print(f"Booting up Production ViT Engine on {DEVICE}...")
    
    dataset = ViTCellDataset(DATA_DIR)
    train_size = 400
    val_size = len(dataset) - train_size
    
    torch.manual_seed(42)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = SimpleViT(patch_size=PATCH_SIZE, embed_dim=EMBED_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    
    # FIX 3: AdamW Optimizer with Weight Decay to stabilize the Transformer loss landscape
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    history_train_loss = []
    history_val_acc = []
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            preds = model(imgs)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        history_train_loss.append(avg_loss)
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                preds = model(imgs)
                predictions = torch.argmax(preds, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
        val_acc = (correct / total) * 100
        history_val_acc.append(val_acc)
        print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.2f}%")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_train_loss, label='Train Loss', color='red')
    plt.title("Production ViT Loss Curve")
    plt.xlabel("Epoch")
    
    plt.subplot(1, 2, 2)
    plt.plot(history_val_acc, label='Val Accuracy', color='blue')
    plt.title("Production ViT Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    
    plt.tight_layout()
    plt.savefig("vit_training_curve.png")
    print("\nRun complete. Plot updated at 'vit_training_curve.png'.")

if __name__ == "__main__":
    train_vit()