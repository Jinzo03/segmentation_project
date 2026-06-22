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
LATENT_DIM = 100
EMBED_DIM = 10     # Dimensional size of the label conditioning embedding vector
NUM_CLASSES = 2    # Class 0: Small Nuclei, Class 1: Large Cells
IMG_CHANNELS = 1
BATCH_SIZE = 64
EPOCHS = 20
OUTPUT_DIR = "cgan_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Label-Conditioned Dataset Builder
class ConditionedCellDataset(Dataset):
    """Generates explicit small vs large cells on the fly with accompanying class labels."""
    def __init__(self, num_samples=2500):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        canvas = np.zeros((64, 64), dtype=np.float32)
        cx, cy = np.random.randint(28, 36), np.random.randint(28, 36)
        
        # Determine cell class on the fly
        label = np.random.choice([0, 1])
        if label == 0:
            radius = np.random.randint(6, 11)   # Explicitly small target radius
            num_pts = 8
        else:
            radius = np.random.randint(18, 25)  # Explicitly large target radius
            num_pts = 12
        
        # Generate contour bounds based on assigned class parameters
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        pts = []
        for angle in angles:
            r = radius + np.random.randint(-2, 3) if label == 0 else radius + np.random.randint(-4, 5)
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            pts.append([x, y])
            
        cv2.fillPoly(canvas, [np.array(pts, dtype=np.int32)], 1.0)
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        
        canvas = (canvas * 2.0) - 1.0  # Normalize to [-1, 1] range
        return torch.tensor(canvas).unsqueeze(0), torch.tensor(label, dtype=torch.long)

# 2. The Conditional Generator Architecture
class ConditionalGenerator(nn.Module):
    def __init__(self):
        super(ConditionalGenerator, self).__init__()
        # Continuous embedding space to convert discrete class IDs to compact vectors
        self.label_embed = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        
        self.net = nn.Sequential(
            # Input dimension is expanded: Latent Noise + Embedded Label Dimension (100 + 10 = 110)
            nn.ConvTranspose2d(LATENT_DIM + EMBED_DIM, 256, 4, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(32, IMG_CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Format labels into a spatial 4D channel block (Batch, EMBED_DIM, 1, 1)
        label_vector = self.label_embed(labels).unsqueeze(2).unsqueeze(3)
        # Concatenate noise vectors alongside the conditioning channel
        x = torch.cat([noise, label_vector], dim=1)
        return self.net(x)

# 3. The Conditional Discriminator Architecture
class ConditionalDiscriminator(nn.Module):
    def __init__(self):
        super(ConditionalDiscriminator, self).__init__()
        # Map class labels directly to a flat channel mask matching image spatial resolution (64x64)
        self.label_embed = nn.Embedding(NUM_CLASSES, 64 * 64)
        
        self.net = nn.Sequential(
            # Input channels: Raw Image Channel + 1 Conditioning Mask Channel = 2 Total Channels
            nn.Conv2d(IMG_CHANNELS + 1, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Flatten()
        )

    def forward(self, img, labels):
        # Format scalar target label identifiers to full spatial feature masks (Batch, 1, 64, 64)
        label_mask = self.label_embed(labels).view(-1, 1, 64, 64)
        # Combine image canvas and class tracking condition across channel matrices
        x = torch.cat([img, label_mask], dim=1)
        return self.net(x)

def save_conditional_grid(generator, epoch):
    """Generates a structured grid: Row 1 forced to be Small Cells, Row 2 forced to be Large Cells."""
    generator.eval()
    # Continuous fixed anchor noise to trace morphing properties explicitly
    fixed_noise = torch.randn(4, LATENT_DIM, 1, 1, device=DEVICE).repeat(2, 1, 1, 1)
    
    # 4 items forced to Class 0 (Small), 4 items forced to Class 1 (Large)
    fixed_labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        fake_imgs = generator(fixed_noise, fixed_labels).cpu().numpy()
    generator.train()
    
    fake_imgs = (fake_imgs + 1.0) / 2.0  # Denormalize
    
    fig, axes = plt.subplots(2, 4, figsize=(8, 4))
    for idx, ax in enumerate(axes.flatten()):
        ax.imshow(fake_imgs[idx, 0], cmap="gray")
        ax.axis("off")
        if idx < 4:
            ax.set_title("Cmd: Small", fontsize=9, color="blue")
        else:
            ax.set_title("Cmd: Large", fontsize=9, color="darkred")
            
    plt.suptitle(f"Conditional GAN Calibration - Epoch {epoch}", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"epoch_{epoch:02d}.png"))
    plt.close()

def train_cgan():
    print(f"Booting Up Conditional DCGAN Core Engine on device: {DEVICE}...")
    dataset = ConditionedCellDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    
    netG = ConditionalGenerator().to(DEVICE)
    netD = ConditionalDiscriminator().to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    for epoch in range(1, EPOCHS + 1):
        loss_d_cum, loss_g_cum = 0.0, 0.0
        
        for real_imgs, labels in dataloader:
            real_imgs, labels = real_imgs.to(DEVICE), labels.to(DEVICE)
            curr_batch_size = real_imgs.size(0)
            
            # ---------------------------------------------------
            # Step A: Train Discriminator (Evaluate Real vs Fake + Label Matching)
            # ---------------------------------------------------
            netD.zero_grad()
            
            # Test Real Images combined with matching Real Labels
            labels_real = torch.ones(curr_batch_size, 1, device=DEVICE)
            output_real = netD(real_imgs, labels)
            loss_d_real = criterion(output_real, labels_real)
            
            # Test Fake Images combined with matching Fake Labels
            noise = torch.randn(curr_batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            fake_labels = torch.from_numpy(np.random.choice([0, 1], size=curr_batch_size)).long().to(DEVICE)
            fake_imgs = netG(noise, fake_labels)
            
            labels_fake = torch.zeros(curr_batch_size, 1, device=DEVICE)
            output_fake = netD(fake_imgs.detach(), fake_labels)
            loss_d_fake = criterion(output_fake, labels_fake)
            
            loss_D = loss_d_real + loss_d_fake
            loss_D.backward()
            optimizerD.step()
            loss_d_cum += loss_D.item()
            
            # ---------------------------------------------------
            # Step B: Train Generator (Fool Discriminator on Specified Class)
            # ---------------------------------------------------
            netG.zero_grad()
            output_fake_for_g = netD(fake_imgs, fake_labels)
            loss_G = criterion(output_fake_for_g, labels_real)
            loss_G.backward()
            optimizerG.step()
            loss_g_cum += loss_G.item()
            
        print(f"Epoch [{epoch}/{EPOCHS}] | Loss D: {loss_d_cum/len(dataloader):.4f} | Loss G: {loss_g_cum/len(dataloader):.4f}")
        save_conditional_grid(netG, epoch)
        
    print("\nConditional GAN training suite fully processed!")

if __name__ == "__main__":
    train_cgan()