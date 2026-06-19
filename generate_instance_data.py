import os
import cv2
import numpy as np

# Configuration
OUTPUT_DIR = "data_instance"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
ANNO_DIR = os.path.join(OUTPUT_DIR, "annotations")
NUM_IMAGES = 100  # Start with 100 images for a fast sanity check run
IMG_SIZE = 256

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(ANNO_DIR, exist_ok=True)

def generate_dataset():
    print(f"Generating {NUM_IMAGES} instance segmentation images...")
    
    for i in range(NUM_IMAGES):
        # Create a black background canvas for the RGB image
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        
        # Determine a random number of objects for this specific image (2 to 5)
        num_objects = np.random.randint(2, 6)
        
        boxes = []
        labels = []
        masks = []
        
        for _ in range(num_objects):
            # Create an isolated black mask for this specific single object instance
            inst_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            
            # Randomly pick object type: 1 = Circle, 2 = Square (0 is background)
            obj_type = np.random.randint(1, 3) 
            
            # Randomize size and coordinates
            size = np.random.randint(25, 55)
            cx = np.random.randint(size, IMG_SIZE - size)
            cy = np.random.randint(size, IMG_SIZE - size)
            
            if obj_type == 1:  # Circle
                # Draw on the main RGB image (Red)
                cv2.circle(img, (cx, cy), size, (255, 0, 0), -1)
                # Draw on the isolated instance mask
                cv2.circle(inst_mask, (cx, cy), size, 1, -1)
                labels.append(1)
                
            elif obj_type == 2:  # Square
                # Draw on the main RGB image (Blue)
                cv2.rectangle(img, (cx - size, cy - size), (cx + size, cy + size), (0, 0, 255), -1)
                # Draw on the isolated instance mask
                cv2.rectangle(inst_mask, (cx - size, cy - size), (cx + size, cy + size), 1, -1)
                labels.append(2)
                
            # Compute the bounding box coordinates [xmin, ymin, xmax, ymax] from the instance mask
            pos = np.where(inst_mask > 0)
            ymin, xmin = np.min(pos[0]), np.min(pos[1])
            ymax, xmax = np.max(pos[0]), np.max(pos[1])
            
            # Mask R-CNN throws errors if boxes have zero width or height
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                masks.append(inst_mask)
            else:
                # Remove label if object drawing failed boundaries
                labels.pop()

        # Handle edge cases where no valid objects were registered
        if len(boxes) == 0:
            continue
            
        # Convert metadata lists to structured NumPy arrays
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        masks = np.array(masks, dtype=np.uint8) # Shape: (N, 256, 256)
        
        # Save files using matching base indexes
        base_name = f"shape_{i:04d}"
        
        # Save the primary image
        cv2.imwrite(os.path.join(IMAGES_DIR, f"{base_name}.png"), img)
        
        # Compress and save annotations (boxes, labels, and 3D multi-layer masks) together
        np.savez_compressed(
            os.path.join(ANNO_DIR, f"{base_name}.npz"),
            boxes=boxes,
            labels=labels,
            masks=masks
        )

    print("Data generation complete!")

if __name__ == "__main__":
    generate_dataset()