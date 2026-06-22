import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SYNTHETIC_SAMPLES = 500  # Number of perfect image/mask pairs to manufacture
OUTPUT_DIR = "synthetic_dataset"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
MASKS_DIR = os.path.join(OUTPUT_DIR, "masks")

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

# Define U-Net components exactly matching your trained architecture
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
        return torch.cat((x, skip_input), 1)

class UNetGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = UNetDown(1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512)
        self.up1 = UNetUp(512, 256)
        self.up2 = UNetUp(512, 128)
        self.up3 = UNetUp(256, 64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
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

def generate_procedural_layout():
    """
    Simulation Engine: Generates highly complex, overlapping, and touching 
    cell layouts to challenge downstream segmentation networks.
    """
    mask = np.zeros((64, 64), dtype=np.float32)
    
    # Randomly decide to place between 1 and 3 cells in this specific patch region
    num_cells = np.random.randint(1, 4)
    
    for _ in range(num_cells):
        # Allow cells to overlap by drifting centers near each other
        cx = np.random.randint(20, 44)
        cy = np.random.randint(20, 44)
        radius = np.random.randint(10, 18)
        
        num_pts = np.random.randint(8, 13)
        angles = np.linspace(0, 2 * np.pi, num_pts, endpoint=False)
        pts = []
        for angle in angles:
            r = radius + np.random.randint(-3, 4)
            pts.append([int(cx + r * np.cos(angle)), int(cy + r * np.sin(angle))])
            
        # Draw onto our clean, pixel-perfect ground truth mask
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1.0)
        
    return mask

def run_data_factory():

    print(f"Initializing Data Factory Core Engine...")
    
    # 1. Instantiate Generator
    generator = UNetGenerator().to(DEVICE)
    
    # === UNCOMMENT AND ACTIVATE WEIGHT LOADING HERE ===
    if os.path.exists("pix2pix_generator.pth"):
        generator.load_state_dict(torch.load("pix2pix_generator.pth", map_location=DEVICE))
        print(" -> Successfully loaded trained Pix2Pix weights!")
    else:
        print(" -> WARNING: No trained weights found! Running with random initialization.")
        
    generator.eval()
    # ... rest of the script remains exactly the same ...
    
    print(f"Manufacturing {NUM_SYNTHETIC_SAMPLES} hyper-realistic samples...")
    
    for i in range(1, NUM_SYNTHETIC_SAMPLES + 1):
        # Step A: Run the procedural simulator to get a clean vector mask
        raw_mask = generate_procedural_layout()
        
        # Format mask tensor to feed into Pix2Pix pipeline: scale to [-1, 1]
        mask_tensor = (torch.tensor(raw_mask).unsqueeze(0).unsqueeze(0) * 2.0) - 1.0
        mask_tensor = mask_tensor.to(DEVICE)
        
        # Step B: Run it through the Pix2Pix rendering engine
        with torch.no_grad():
            synthetic_render = generator(mask_tensor).cpu().squeeze().numpy()
            
        # Denormalize both back to standard image ranges [0, 255]
        final_image = ((synthetic_render + 1.0) / 2.0 * 255.0).astype(np.uint8)
        final_mask = (raw_mask * 255.0).astype(np.uint8)
        
        # Save to their respective clean dataset repositories
        cv2.imwrite(os.path.join(IMAGES_DIR, f"synth_cell_{i:04d}.png"), final_image)
        cv2.imwrite(os.path.join(MASKS_DIR, f"synth_mask_{i:04d}.png"), final_mask)
        
        if i % 100 == 0:
            print(f" -> Manufactured {i}/{NUM_SYNTHETIC_SAMPLES} image pairs.")
            
    # Save a visual inspection sheet of the factory output
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(final_mask, cmap="gray")
    axes[0].set_title("Auto-Labeled Mask")
    axes[0].axis("off")
    
    axes[1].imshow(final_image, cmap="gray")
    axes[1].set_title("AI Textured Image")
    axes[1].axis("off")
    
    plt.suptitle("Factory Visual Quality Assurance Check")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "factory_qa_sample.png"))
    plt.close()

    print(f"\nSuccess! Dataset completely rendered at: /{OUTPUT_DIR}")
    print("Ready to link directly to downstream training pipelines.")

if __name__ == "__main__":
    run_data_factory()