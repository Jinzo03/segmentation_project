import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "instance_checkpoint_10.pth.tar"
NUM_CLASSES = 3  # 0: Background, 1: Circle, 2: Square
TEST_IMG_DIR = "data_instance/images/"
CONFIDENCE_THRESHOLD = 0.70  # Only display objects the model is more than 70% sure about

def get_instance_model(num_classes):
    """Initializes the baseline architecture framework."""
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def visual_inference_instance():
    # 1. Initialize model architecture and map weights
    print("Loading model and weights...")
    model = get_instance_model(num_classes=NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 2. Grab the first image from the directory
    images = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith('.png')])
    if not images:
        print("No test images found.")
        return
    
    test_file = images[0]
    img_path = os.path.join(TEST_IMG_DIR, test_file)
    print(f"Processing target test image: {test_file}")
    
    # 3. Preprocess the image for execution
    raw_image = Image.open(img_path).convert("RGB")
    img_np = np.array(raw_image)
    
    # Convert image to float tensor scaled between 0 and 1
    input_tensor = torch.from_numpy(img_np.transpose((2, 0, 1))).float() / 255.0
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)

    # 4. Execute Forward Pass
    with torch.no_grad():
        predictions = model(input_tensor)[0]

    # 5. Extract prediction outputs
    boxes = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    masks = predictions["masks"].cpu().numpy()  # Shape: (N, 1, H, W)

    # Filter outputs using confidence score threshold
    keep_idx = np.where(scores > CONFIDENCE_THRESHOLD)[0]
    
    # 6. Initialize plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left Panel: Original Image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image Input")
    axes[0].axis("off")
    
    # Right Panel: Instance Predictions Map
    axes[1].imshow(img_np)
    axes[1].set_title(f"Mask R-CNN Instance Detection (Threshold: {CONFIDENCE_THRESHOLD})")
    axes[1].axis("off")

    # Define strings mapping class labels
    class_map = {1: "Circle", 2: "Square"}
    
    # Generate a set of unique colors to differentiate instances
    cmap = plt.get_cmap("tab10")

    print(f"Detected {len(keep_idx)} distinct object instances above threshold.")

    # 7. Iterate through and overlay each valid instance detection
    h, w, _ = img_np.shape
    for idx, obj_i in enumerate(keep_idx):
        box = boxes[obj_i]
        label = labels[obj_i]
        score = scores[obj_i]
        mask = masks[obj_i, 0]  # Grab the 2D probability grid for this instance
        
        # Binary thresholding for the mask slice
        binary_mask = mask > 0.5
        
        # Pick a distinct color for this unique instance
        color = np.array(cmap(idx % 10)[:3])
        
        # FIX: Create a 4-channel RGBA float array (0.0 to 1.0) natively handled by Matplotlib
        rgba_mask = np.zeros((h, w, 4), dtype=np.float32)
        rgba_mask[binary_mask, :3] = color   # Assign RGB channels
        rgba_mask[binary_mask, 3] = 0.4      # Assign Alpha transparency (40% opacity)
        
        # Overlay the transparent instance mask layer directly on the Right Plot
        axes[1].imshow(rgba_mask)
        
        # Draw the bounding box
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=2, edgecolor=cmap(idx % 10), facecolor='none'
        )
        axes[1].add_patch(rect)
        
        # Annotate text labels above the bounding box
        label_text = f"{class_map.get(label, 'Unknown')}: {score:.2f}"
        axes[1].text(
            xmin, ymin - 4 if ymin - 4 > 10 else ymin + 15, label_text,
            color='white', fontsize=10, weight='bold',
            bbox=dict(facecolor=cmap(idx % 10), alpha=0.8, pad=1, edgecolor='none')
        )

    plt.tight_layout()
    output_filename = "instance_segmentation_output.png"
    plt.savefig(output_filename, dpi=150)
    print(f"Inference complete! Verification plot saved as '{output_filename}'")

if __name__ == "__main__":
    visual_inference_instance()