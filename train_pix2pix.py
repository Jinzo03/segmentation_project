import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 15
LAMBDA_L1 = 100  # Weight factor balance favoring pixel structure matching
OUTPUT_DIR = "pix2pix_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Paired Image-to-Image Dataset Generator
class Pix2PixCellDataset(Dataset):
    """Generates explicit matching pairs: [Binary Mask (Domain A)] -> [Realistic Cell (Domain B)]"""
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Establish blank canvases
        mask = np.zeros((64, 64), dtype=np.float32)
        real_cell = np.zeros((64, 64), dtype=np.float32)
        
        cx, cy = np.random.randint(24, 40), np.random.randint(24, 40)
        radius = np.random.randint(14, 22)
        
        # Draw base shared shape
        num_pts = 10
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        pts = []
        for angle in angles:
            r = radius + np.random.randint(-3, 4)
            pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
            
        # Domain A: Pure sharp binary mask boundaries
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1.0)
        
        # Domain B: Advanced photorealistic translation (texture, lighting gradients, noise)
        cv2.fillPoly(real_cell, [np.array(pts, dtype=np.int32)], 0.8)
        # Add internal texture variations (simulated cellular organelles)
        for _ in range(5):
            tx = cx + np.random.randint(-8, 9)
            ty = cy + np.random.randint(-8, 9)
            cv2.circle(real_cell, (tx, ty), np.random.randint(2, 5), 0.4, -1)
            
        real_cell = cv2.GaussianBlur(real_cell, (5, 5), 0)
        # Apply illumination microscope background gradient
        y_grid, x_grid = np.mgrid[0:64, 0:64]
        gradient = (x_grid + y_grid) / 128.0 * 0.15
        real_cell += gradient
        # Introduce lens sensor grain noise
        real_cell += np.random.normal(0, 0.03, real_cell.shape)
        real_cell = np.clip(real_cell, 0.0, 1.0)
        
        # Scale both to [-1, 1] range standard optimization
        mask = (mask * 2.0) - 1.0
        real_cell = (real_cell * 2.0) - 1.0
        
        return torch.tensor(mask).unsqueeze(0), torch.tensor(real_cell).unsqueeze(0)

# 2. Reusable Blocks for Generator / Discriminator
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), 1) # Spatial Skip Connection binding

# 3. The U-Net Generator Network
class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = UNetDown(1, 64, normalize=False) # 64x64 -> 32x32
        self.down2 = UNetDown(64, 128)                # 32x32 -> 16x16
        self.down3 = UNetDown(128, 256)               # 16x16 -> 8x8
        self.down4 = UNetDown(256, 512)               # 8x8 -> 4x4
        
        self.up1 = UNetUp(512, 256)                   # 4x4 -> 8x8 + skip
        self.up2 = UNetUp(512, 128)                   # 8x8 -> 16x16 + skip
        self.up3 = UNetUp(256, 64)                    # 16x16 -> 32x32 + skip
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, 2, 1),      # 32x32 -> 64x64
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        
        u1 = self.up1(d4, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        return self.final(u3)

# 4. The PatchGAN Discriminator Network
class PatchGANDiscriminator(nn.Module):
    """Evaluates patches of combined input mapping to judge localized realism structures."""
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Takes Mask + Target Image concatenated together = 2 channels
            nn.Conv2d(2, 64, 4, 2, 1), # 64x64 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), # 32x32 -> 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False), # 16x16 -> 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, 4, 1, 0) # Output Patch matrix evaluation maps
        )

    def forward(self, mask, img):
        # Concatenate condition mask and evaluated target across channels
        x = torch.cat((mask, img), 1)
        return self.model(x)

def save_pix2pix_progress(generator, src_mask, target_real, epoch):
    """Plots visual comparison row: Source Mask | Ground Truth Real Image | AI Translated Output"""
    generator.eval()
    with torch.no_grad():
        fake_target = generator(src_mask).cpu().numpy()
    generator.train()
    
    src_img = (src_mask[0, 0].cpu().numpy() + 1.0) / 2.0
    real_img = (target_real[0, 0].cpu().numpy() + 1.0) / 2.0
    translated_img = (fake_target[0, 0] + 1.0) / 2.0
    
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    axes[0].imshow(src_img, cmap="gray")
    axes[0].set_title("Input Binary Mask")
    axes[0].axis("off")
    
    axes[1].imshow(real_img, cmap="gray")
    axes[1].set_title("Target Real Microscope")
    axes[1].axis("off")
    
    axes[2].imshow(translated_img, cmap="gray")
    axes[2].set_title("AI Rendered Translation")
    axes[2].axis("off")
    
    plt.suptitle(f"Pix2Pix Translation Sync - Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"epoch_{epoch:02d}.png"))
    plt.close()

def train_pix2pix():
    print(f"Ignition sequence: Training Pix2Pix Engine on device: {DEVICE}...")
    dataset = Pix2PixCellDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    netG = UNetGenerator().to(DEVICE)
    netD = PatchGANDiscriminator().to(DEVICE)
    
    gan_loss_fn = nn.BCEWithLogitsLoss()
    l1_loss_fn = nn.L1Loss()
    
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Extract fixed anchor pair to visually trace calibration stability over training timeline
    fixed_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    fixed_mask, fixed_real = next(iter(fixed_dataloader))
    fixed_mask, fixed_real = fixed_mask.to(DEVICE), fixed_real.to(DEVICE)
    
    for epoch in range(1, EPOCHS + 1):
        loss_d_cum, loss_g_cum = 0.0, 0.0
        
        for masks, real_imgs in dataloader:
            masks, real_imgs = masks.to(DEVICE), real_imgs.to(DEVICE)
            
            # ---------------------------------------------------
            # Step A: Train PatchGAN Discriminator
            # ---------------------------------------------------
            netD.zero_grad()
            
            # Evaluate Real Pair
            pred_real = netD(masks, real_imgs)
            patch_shape = pred_real.shape
            labels_real = torch.ones(patch_shape, device=DEVICE)
            loss_d_real = gan_loss_fn(pred_real, labels_real)
            
            # Evaluate Generated Fake Pair
            fake_imgs = netG(masks)
            pred_fake = netD(masks, fake_imgs.detach())
            labels_fake = torch.zeros(patch_shape, device=DEVICE)
            loss_d_fake = gan_loss_fn(pred_fake, labels_fake)
            
            loss_D = (loss_d_real + loss_d_fake) * 0.5
            loss_D.backward()
            optimizerD.step()
            loss_d_cum += loss_D.item()
            
            # ---------------------------------------------------
            # Step B: Train U-Net Generator (Adversarial Realism + Strict L1 Structural Loss)
            # ---------------------------------------------------
            netG.zero_grad()
            
            pred_fake_for_g = netD(masks, fake_imgs)
            loss_gan = gan_loss_fn(pred_fake_for_g, labels_real) # Fool the detective patch
            loss_l1 = l1_loss_fn(fake_imgs, real_imgs)          # Exact pixel alignments
            
            loss_G = loss_gan + (LAMBDA_L1 * loss_l1)
            loss_G.backward()
            optimizerG.step()
            loss_g_cum += loss_G.item()
            
        print(f"Epoch [{epoch}/{EPOCHS}] | Loss PatchGAN D: {loss_d_cum/len(dataloader):.4f} | Loss U-Net G: {loss_g_cum/len(dataloader):.4f}")
        save_pix2pix_progress(netG, fixed_mask, fixed_real, epoch)
        
    print("\nPix2Pix translation calibration absolute. Render complete!")
    # === ADD THIS LINE AT THE END OF train_pix2pix() ===
    torch.save(netG.state_dict(), "pix2pix_generator.pth")
    print("Model weights successfully serialized to disk!")
    
if __name__ == "__main__":
    train_pix2pix()