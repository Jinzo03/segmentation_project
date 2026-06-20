import os
import cv2
import numpy as np

OUTPUT_DIR = "cell_video"
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_SIZE = 256
NUM_FRAMES = 30

def generate_organic_blob(cx, cy, base_radius, seed):
    """Generates an irregular cell contour based on a static seed value to keep shape consistency."""
    np.random.seed(seed)
    num_points = 12
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = []
    for angle in angles:
        r = base_radius + np.random.randint(-4, 5)
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))
        points.append([x, y])
    return np.array(points, dtype=np.int32)

def generate_video_sequence():
    print("Generating 30-frame simulated live cell video...")
    
    # Define 5 cells with initial positions, static shapes, and constant velocities
    np.random.seed(42)
    cells = []
    for idx in range(5):
        cells.append({
            "cx": np.random.randint(40, IMG_SIZE - 40),
            "cy": np.random.randint(40, IMG_SIZE - 40),
            "radius": np.random.randint(18, 26),
            "vx": np.random.choice([-3, -2, 2, 3]), # X velocity component
            "vy": np.random.choice([-3, -2, 2, 3]), # Y velocity component
            "seed": idx * 100,
            "color": [np.random.randint(130, 160), np.random.randint(50, 70), np.random.randint(140, 180)]
        })

    for frame_idx in range(NUM_FRAMES):
        # Create canvas with microscope background artifacts
        img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        X, Y = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE))
        distance = np.sqrt((X - 128)**2 + (Y - 128)**2)
        img += (25 * (1.0 - distance / np.max(distance)))[:, :, None] + 20
        
        cell_composite = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        
        # Update and draw each cell frame-by-frame
        for c in cells:
            # Advance spatial positions linearly
            c["cx"] += c["vx"]
            c["cy"] += c["vy"]
            
            # Simple bounce-back boundary conditions
            if c["cx"] < c["radius"] or c["cx"] > IMG_SIZE - c["radius"]: c["vx"] *= -1
            if c["cy"] < c["radius"] or c["cy"] > IMG_SIZE - c["radius"]: c["vy"] *= -1
            
            pts = generate_organic_blob(c["cx"], c["cy"], c["radius"], c["seed"])
            
            cell_layer = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            cv2.fillPoly(cell_layer, [pts], c["color"])
            cell_layer = cv2.GaussianBlur(cell_layer, (5, 5), 0)
            
            mask_bool = np.any(cell_layer > 0, axis=2)
            cell_composite[mask_bool] = cell_composite[mask_bool] * 0.3 + cell_layer[mask_bool] * 0.7

        img += cell_composite
        noise = np.random.normal(0, 2.0, img.shape)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Save sequential frame
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{frame_idx:04d}.png"), img)
        
    print(f"Video data successfully cooked under '{OUTPUT_DIR}/' directory!")

if __name__ == "__main__":
    generate_video_sequence()