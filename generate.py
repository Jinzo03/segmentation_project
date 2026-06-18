import os
import numpy as np
from PIL import Image, ImageDraw

def generate_toy_data(num_samples=50, size=256):
    os.makedirs("toy_data/images", exist_ok=True)
    os.makedirs("toy_data/masks", exist_ok=True)
    
    for i in range(num_samples):
        # Create blank image
        img = Image.new('RGB', (size, size), (0, 0, 0))
        mask = Image.new('L', (size, size), 0)
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)
        
        # Draw random circle
        x, y = np.random.randint(50, 200, 2)
        r = np.random.randint(20, 50)
        
        draw_img.ellipse((x-r, y-r, x+r, y+r), fill=(255, 255, 255))
        draw_mask.ellipse((x-r, y-r, x+r, y+r), fill=255)
        
        img.save(f"toy_data/images/{i}.png")
        mask.save(f"toy_data/masks/{i}.png")

generate_toy_data()