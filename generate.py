import os
import numpy as np
from PIL import Image, ImageDraw

def generate_multiclass_data(num_samples=100, size=256):
    os.makedirs("data/images", exist_ok=True)
    os.makedirs("data/masks", exist_ok=True)
    
    for i in range(num_samples):
        img = Image.new('RGB', (size, size), (0, 0, 0))
        mask = Image.new('L', (size, size), 0) # Background is 0
        draw_img = ImageDraw.Draw(img)
        draw_mask = ImageDraw.Draw(mask)
        
        # Draw Circle (Class 1)
        c_x, c_y = np.random.randint(50, 200, 2)
        c_r = np.random.randint(20, 40)
        draw_img.ellipse((c_x-c_r, c_y-c_r, c_x+c_r, c_y+c_r), fill=(255, 0, 0))
        draw_mask.ellipse((c_x-c_r, c_y-c_r, c_x+c_r, c_y+c_r), fill=1)
        
        # Draw Square (Class 2)
        s_x, s_y = np.random.randint(50, 200, 2)
        s_s = np.random.randint(20, 40)
        draw_img.rectangle((s_x, s_y, s_x+s_s*2, s_y+s_s*2), fill=(0, 0, 255))
        draw_mask.rectangle((s_x, s_y, s_x+s_s*2, s_y+s_s*2), fill=2)
        
        img.save(f"data/images/{i}.png")
        mask.save(f"data/masks/{i}.png")

generate_multiclass_data()