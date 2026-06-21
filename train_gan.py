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
LATENT_DIM = 100  # Size of the random noise vector input
IMG_CHANNELS = 1  # Grayscale cell templates for maximum training stability
BATCH_SIZE = 64
EPOCHS = 20
OUTPUT_DIR = "gan_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. On-the-Fly Cell Dataset Generator
class SyntheticCellDataset(Dataset):
    """Generates ground-truth cell imagery on the fly for the GAN to learn from."""
    def __init__(self, num_samples=2000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create a black 64x64 canvas
        canvas = np.zeros((64, 64), dtype=np.float32)
        cx, cy = np.random.randint(24, 40), np.random.randint(24, 40)
        radius = np.random.randint(12, 20)
        
        # Generate an organic blob contour
        num_pts = 10
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        pts = []
        for angle in angles:
            r = radius + np.random.randint(-3, 4)
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            pts.append([x, y])
            
        cv2.fillPoly(canvas, [np.array(pts, dtype=np.int32)], 1.0)
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        
        # Scale to [-1, 1] range (crucial standard optimization for GAN Tanh outputs)
        canvas = (canvas * 2.0) - 1.0
        return torch.tensor(canvas).unsqueeze(0)

# 2. The Generator Network Architecture
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: Latent noise vector layer (Batch, 100, 1, 1)
            nn.ConvTranspose2d(LATENT_DIM, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # State: (Batch, 256, 4, 4)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # State: (Batch, 128, 8, 8)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # State: (Batch, 64, 16, 16)
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            # State: (Batch, 32, 32, 32) -> Output: (Batch, 1, 64, 64)
            nn.ConvTranspose2d(32, IMG_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()  # Squashes output pixels directly between -1 and 1
        )

    def forward(self, x):
        return self.net(x)

# 3. The Discriminator Network Architecture
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            # Input image shape: (Batch, 1, 64, 64)
            nn.Conv2d(IMG_CHANNELS, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (Batch, 32, 32, 32)
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (Batch, 64, 16, 16)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (Batch, 128, 8, 8)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State: (Batch, 256, 4, 4) -> Output scalar prediction logit
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)

def save_progress_grid(generator, epoch, fixed_noise):
    """Generates images using fixed noise to track structural evolution visually."""
    generator.eval()
    with torch.no_grad():
        fake_imgs = generator(fixed_noise).cpu().numpy()
    generator.train()
    
    # Rescale back from [-1, 1] to [0, 1] display range
    fake_imgs = (fake_imgs + 1.0) / 2.0
    
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for idx, ax in enumerate(axes.flatten()):
        ax.imshow(fake_imgs[idx, 0], cmap="gray")
        ax.axis("off")
    plt.suptitle(f"GAN Generated Outputs - Epoch {epoch}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"epoch_{epoch:02d}.png"))
    plt.close()

def train_gan():
    print(f"Initializing DCGAN Training Pipeline on device: {DEVICE}...")
    dataset = SyntheticCellDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    netG = Generator().to(DEVICE)
    netD = Discriminator().to(DEVICE)
    
    # Binary Cross Entropy loss function with built-in numerical logit stability
    criterion = nn.BCEWithLogitsLoss()
    
    # Standard hyperparameter tunes for stable DCGAN alignment
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # Establish a static noise anchor to trace generation fidelity shifts over time
    fixed_noise = torch.randn(8, LATENT_DIM, 1, 1, device=DEVICE)
    
    for epoch in range(1, EPOCHS + 1):
        loss_d_cum = 0.0
        loss_g_cum = 0.0
        
        for real_imgs in dataloader:
            real_imgs = real_imgs.to(DEVICE)
            curr_batch_size = real_imgs.size(0)
            
            # ---------------------------------------------------
            # Step A: Train the Discriminator (Maximize Real vs Fake Separation)
            # ---------------------------------------------------
            netD.zero_grad()
            
            # Test Real Images
            labels_real = torch.ones(curr_batch_size, 1, device=DEVICE)
            output_real = netD(real_imgs)
            loss_d_real = criterion(output_real, labels_real)
            
            # Test Generated Fake Images
            noise = torch.randn(curr_batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_imgs = netG(noise)
            labels_fake = torch.zeros(curr_batch_size, 1, device=DEVICE)
            output_fake = netD(fake_imgs.detach()) # Detach prevents gradients leaking into G
            loss_d_fake = criterion(output_fake, labels_fake)
            
            loss_D = loss_d_real + loss_d_fake
            loss_D.backward()
            optimizerD.step()
            loss_d_cum += loss_D.item()
            
            # ---------------------------------------------------
            # Step B: Train the Generator (Fool the Discriminator)
            # ---------------------------------------------------
            netG.zero_grad()
            output_fake_for_g = netD(fake_imgs)
            # The Generator wants the Discriminator to believe these fakes are Real (label=1)
            loss_G = criterion(output_fake_for_g, labels_real)
            loss_G.backward()
            optimizerG.step()
            loss_g_cum += loss_G.item()
            
        print(f"Epoch [{epoch}/{EPOCHS}] | Loss D: {loss_d_cum/len(dataloader):.4f} | Loss G: {loss_g_cum/len(dataloader):.4f}")
        save_progress_grid(netG, epoch, fixed_noise)
        
    print("\nDCGAN training phase complete! Progress tracking grids generated.")

if __name__ == "__main__":
    train_gan()