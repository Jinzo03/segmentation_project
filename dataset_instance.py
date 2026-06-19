import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class InstanceSegmentationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to 'data_instance' folder.
            transform (albumentations.Compose): Transforms to apply to images, bboxes, and masks.
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, "images")
        self.anno_dir = os.path.join(root_dir, "annotations")
        self.transform = transform
        
        # Sort files to ensure matching pairs
        self.images = sorted([f for f in os.listdir(self.images_dir) if f.endswith(".png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Load the primary image
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = np.array(Image.open(img_path).convert("RGB"))

        # 2. Load the corresponding compressed .npz annotation file
        base_name = os.path.splitext(img_name)[0]
        anno_path = os.path.join(self.anno_dir, f"{base_name}.npz")
        
        annotation = np.load(anno_path)
        boxes = annotation["boxes"]    # Shape: (N, 4) -> [xmin, ymin, xmax, ymax]
        labels = annotation["labels"]  # Shape: (N,)
        masks = annotation["masks"]    # Shape: (N, H, W)

        # 3. Apply Transformations safely
        # Mask R-CNN requires bounding boxes and masks to shift perfectly alongside image augmentations
        if self.transform:
            # Albumentations expects a list of 2D masks rather than a 3D block
            mask_list = [masks[j] for j in range(masks.shape[0])]
            
            augmented = self.transform(
                image=image,
                masks=mask_list,
                bboxes=boxes,
                category_ids=labels
            )
            image = augmented["image"]
            
            # Reassemble augmented components back into clean arrays
            if len(augmented["masks"]) > 0:
                masks = np.stack(augmented["masks"], axis=0)
                boxes = np.array(augmented["bboxes"], dtype=np.float32)
                labels = np.array(augmented["category_ids"], dtype=np.int64)
            else:
                # Fallback if augmentations accidentally crop/eliminate all shapes
                masks = np.zeros((0, image.shape[1], image.shape[2]), dtype=np.uint8)
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)

        # 4. Convert targets directly into PyTorch Tensors
        # PyTorch Detection expects these specific dictionary key namings
        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.long)
        target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
        
        # Unique identifier requirement for evaluation engines
        target["image_id"] = torch.tensor([idx])

        return image, target

def collate_fn(batch):
    """
    CRITICAL FIX: Overriding the default DataLoader collate function.
    Because different images contain varying numbers of objects (N varies),
    we cannot stack targets into a traditional square tensor. 
    Instead, we bundle them into a clean Python tuple.
    """
    return tuple(zip(*batch))

if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from torch.utils.data import DataLoader

    # Simple pipeline transformation with Bbox definitions
    test_transform = A.Compose([
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    # Instantiate
    dataset = InstanceSegmentationDataset(root_dir="data_instance", transform=test_transform)
    
    # Notice the usage of our custom collate_fn here!
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    # Pull a single batch
    images, targets = next(iter(loader))
    
    print(f"Batch loaded successfully!")
    print(f"Number of images in batch: {len(images)}")
    print(f"Image 0 Tensor Shape: {images[0].shape}")
    print(f"Image 0 detected objects count: {len(targets[0]['labels'])}")
    print(f"Image 0 Boxes Tensor Shape: {targets[0]['boxes'].shape}")
    print(f"Image 0 Masks Tensor Shape: {targets[0]['masks'].shape}")