import os
import cv2
import numpy as np

# Configuration
OUTPUT_DIR = "data_cells"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
ANNO_DIR = os.path.join(OUTPUT_DIR, "annotations")
NUM_IMAGES = 100  # 100 high-density slide frames
IMG_SIZE = 256

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ANNO_DIR, exist_ok=True)

def generate_organic_blob(cx, cy, base_radius):
    """Generates an irregular, wavy organic shape contour."""
    num_points = 12
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = []
    for angle in angles:
        # Mutate radius size continuously to build organic contours
        r = base_radius + np.random.randint(-6, 7)
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))
        points.append([x, y])
    return np.array(points, dtype=np.int32)

def generate_cell_dataset():
    print(f"Generating {NUM_IMAGES} simulated biological cell slides...")
    
    for i in range(NUM_IMAGES):
        # 1. Initialize canvas matrix
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        
        # 2. Add a lighting gradient overlay (uneven slide glare/vignetting)
        X, Y = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
        glare_cx, glare_cy = np.random.randint(0, IMG_SIZE), np.random.randint(0, IMG_SIZE)
        distance = np.sqrt((X - glare_cx)**2 + (Y - glare_cy)**2)
        gradient = 35 * (1.0 - distance / np.max(distance))
        for c in range(3):
            img[:, :, c] += gradient + np.random.randint(15, 30) # Soft grey/brown background tint
            
        # Define high crowding density (4 to 8 cells per slide frame)
        num_cells = np.random.randint(4, 9)
        
        boxes = []
        labels = []
        masks = []
        
        cell_composite = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        
        for _ in range(num_cells):
            inst_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            
            radius = np.random.randint(16, 30)
            cx = np.random.randint(radius, IMG_SIZE - radius)
            cy = np.random.randint(radius, IMG_SIZE - radius)
            
            pts = generate_organic_blob(cx, cy, radius)
            
            # Draw precise ground-truth mask component
            cv2.fillPoly(inst_mask, [pts], 1)
            
            # Generate hematoxylin-stained cell color layers (shades of dark blue/violet)
            cell_layer = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            color = [np.random.randint(120, 170), np.random.randint(40, 75), np.random.randint(130, 190)]
            cv2.fillPoly(cell_layer, [pts], color)
            
            # Apply blur to simulate fuzzy cellular boundaries / lens out-of-focus layers
            blur_k = np.random.choice([3, 5, 7])
            cell_layer = cv2.GaussianBlur(cell_layer, (blur_k, blur_k), 0)
            
            # Merge cell layer onto the running composite canvas with translucent blending
            mask_bool = np.any(cell_layer > 0, axis=2)
            cell_composite[mask_bool] = cell_composite[mask_bool] * 0.2 + cell_layer[mask_bool] * 0.8
            
            pos = np.where(inst_mask > 0)
            if len(pos[0]) > 0 and len(pos[1]) > 0:
                ymin, xmin = np.min(pos[0]), np.min(pos[1])
                ymax, xmax = np.max(pos[0]), np.max(pos[1])
                
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    masks.append(inst_mask)
                    labels.append(1)  # Class ID 1 represents 'Cell Nucleus'

        # Add the cell clusters directly to our slide lighting profile
        img += cell_composite
        
        # 3. Inject microscope camera sensor grain/noise
        noise = np.random.normal(0, 3.5, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        if len(boxes) == 0:
            continue
            
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        masks = np.array(masks, dtype=np.uint8)
        
        base_name = f"cell_{i:04d}"
        cv2.imwrite(os.path.join(IMAGES_DIR, f"{base_name}.png"), img)
        np.savez_compressed(
            os.path.join(ANNO_DIR, f"{base_name}.npz"),
            boxes=boxes,
            labels=labels,
            masks=masks
        )
        
    print("Simulated cell dataset complete! Data saved under 'data_cells/'.")

if __name__ == "__main__":
    generate_cell_dataset()