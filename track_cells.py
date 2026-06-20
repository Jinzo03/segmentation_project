import os
import torch
import numpy as np
import cv2
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "instance_checkpoint_10.pth.tar"
VIDEO_DIR = "cell_video"
OUTPUT_DIR = "tracked_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CONF_THRESHOLD = 0.65

def get_instance_model():
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2) # 0: Background, 1: Cell
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, 2)
    return model

def calculate_iou(boxA, boxB):
    """Computes the Intersection over Union metric between two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2] if 'boxB' in locals() else boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    unionArea = boxAArea + boxBArea - interArea
    return interArea / float(unionArea) if unionArea > 0 else 0

def main():
    print("Loading Mask R-CNN network layers...")
    model = get_instance_model().to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    frames = sorted([f for f in os.listdir(VIDEO_DIR) if f.endswith(".png")])
    
    # State tracking managers
    next_id = 1
    tracked_objects = {} # Dictionary holding running state: {id: bounding_box}
    id_colors = {}       # Maintain persistent colors for each active ID
    
    print("Processing video timeline frames...")
    for frame_name in frames:
        frame_path = os.path.join(VIDEO_DIR, frame_name)
        img_raw = cv2.imread(frame_path)
        h, w, _ = img_raw.shape
        
        # Prepare frame for model input
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        input_tensor = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float() / 255.0
        input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            predictions = model(input_tensor)[0]
            
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        masks = predictions["masks"].cpu().numpy()
        
        # Filter raw predictions using confidence threshold
        keep = np.where(scores > CONF_THRESHOLD)[0]
        current_detections = []
        
        for idx in keep:
            current_detections.append({
                "box": boxes[idx],
                "mask": masks[idx, 0] > 0.5
            })
            
        new_tracked_objects = {}
        
        # Data Association Step via Greedy IoU Maximization Match
        for det in current_detections:
            best_id = None
            best_iou = 0.25 # Minimum acceptable IoU match overlap threshold
            
            for obj_id, old_box in tracked_objects.items():
                iou = calculate_iou(det["box"], old_box)
                if iou > best_iou:
                    best_iou = iou
                    best_id = obj_id
            
            if best_id is not None and best_id not in new_tracked_objects:
                # Found a temporal match! Maintain persistent ID
                new_tracked_objects[best_id] = det["box"]
                det["id"] = best_id
            else:
                # No match found. Initialize a brand new cell tracker ID
                new_tracked_objects[next_id] = det["box"]
                det["id"] = next_id
                # Assign a bright tracking color profile
                id_colors[next_id] = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
                next_id += 1

        # Update tracking database register
        tracked_objects = new_tracked_objects
        
        # Rendering Engine: Draw persistent IDs onto the frame output
        for det in current_detections:
            if "id" not in det: continue
            obj_id = det["id"]
            color = id_colors[obj_id]
            box = det["box"].astype(int)
            mask = det["mask"]
            
            # Draw colored translucent cell mask overlay
            mask_layer = np.zeros_like(img_raw, dtype=np.uint8)
            mask_layer[mask] = color
            img_raw = cv2.addWeighted(img_raw, 1.0, mask_layer, 0.4, 0)
            
            # Draw persistent bounding box and ID tag text string
            cv2.rectangle(img_raw, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(img_raw, f"Cell ID: {obj_id}", (box[0], box[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        # Save output tracking frame sequence
        cv2.imwrite(os.path.join(OUTPUT_DIR, frame_name), img_raw)

    print(f"Tracking complete! Outputs saved inside '{OUTPUT_DIR}/'.")

if __name__ == "__main__":
    main()