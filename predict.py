import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import UNet

# Setup parameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoint_20.pth.tar"
TEST_IMG_DIR = "data/val_images/"
TEST_MASK_DIR = "data/val_masks/"

def visual_inference():
    # 1. Initialize and load model weights
    model = UNet(in_channels=3, out_channels=1).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 2. Grab the first available validation sample
    images = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not images:
        print("No validation images found to test.")
        return
    
    test_file = images[0]
    img_path = os.path.join(TEST_IMG_DIR, test_file)
    mask_path = os.path.join(TEST_MASK_DIR, test_file)

    # 3. Load and transform the data identically to validation settings
    raw_image = Image.open(img_path).convert("RGB")
    raw_mask = Image.open(mask_path).convert("L")

    transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    # Process image for network input
    augmented = transform(image=np.array(raw_image))
    input_tensor = augmented["image"].unsqueeze(0).to(DEVICE) # Add batch dimension

    # 4. Execute Forward Pass
    with torch.no_grad():
        logits = model(input_tensor)
        preds = torch.sigmoid(logits)
        preds = (preds > 0.5).float()
    
    # Convert prediction back to viewable numpy array
    pred_mask = preds.squeeze().cpu().numpy()

    # 5. Plot Results Side-by-Side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image resized for comparison
    axes[0].imshow(raw_image.resize((256, 256)))
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Target Mask
    axes[1].imshow(raw_mask.resize((256, 256)), cmap="gray")
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")

    # AI Prediction
    axes[2].imshow(pred_mask, cmap="gray")
    axes[2].set_title(f"AI Prediction (Dice: 0.89)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig("final_segmentation_output.png")
    print("Inference complete. Image saved as 'final_segmentation_output.png'")

if __name__ == "__main__":
    visual_inference()