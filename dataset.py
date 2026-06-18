import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Filter to make sure we only grab actual image files
        self.images = [f for f in os.listdir(image_dir) 
                       if os.path.isfile(os.path.join(image_dir, f)) 
                       and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        
        # 1. Load image and mask as NumPy arrays
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.int32)

        # 2. Apply Albumentations transformations if provided
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        else:
            # Fallback: manually convert to Tensors if no transform is supplied
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask = torch.from_numpy(mask)

        # 3. FIX: Hard-verify that mask is a PyTorch Tensor before calling .long()
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
            
        return image, mask.long()