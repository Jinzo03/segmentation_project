# segmentation_project

A PyTorch pipeline for synthetic biological cell image generation, segmentation, and tracking. It includes an Attention U-Net for semantic/instance segmentation, GAN-based image synthesis (GAN, cGAN, pix2pix), a Vision Transformer training path, and tools to track cells across frames and compile the results into video.

## Features

- **Synthetic data generation** — procedurally generates microscope-style slide images with organic cell blobs, lighting glare, hematoxylin-style staining, blur, and sensor noise, along with matching masks/boxes/labels.
- **Segmentation models** — an Attention U-Net (`model.py`) with attention gates on the skip connections, plus a Vision Transformer training path (`train_vit.py`).
- **Instance segmentation** — separate dataset, training, and prediction scripts for per-cell instance masks.
- **GAN-based synthesis** — GAN, conditional GAN, and pix2pix training scripts for generating or translating cell imagery.
- **Cell tracking & video** — tracks cells across generated frames and compiles results into video for visual inspection.

## Repository structure

```
segmentation_project/
├── data/                        # Generated/raw data
├── run_data_factory.py          # Orchestrates synthetic data generation
├── generate.py                  # General synthetic data generator
├── generate_cell_data.py        # Generates semantic segmentation cell dataset
├── generate_instance_data.py    # Generates instance segmentation cell dataset
├── generate_cell_video.py       # Generates a synthetic video sequence of cells
├── dataset.py                   # PyTorch Dataset for semantic segmentation
├── dataset_instance.py          # PyTorch Dataset for instance segmentation
├── tokenizer.py                 # Patch/tokenization utilities (for ViT)
├── model.py                     # Attention U-Net architecture
├── train.py                     # General training entry point
├── train_segmentation.py        # Trains the segmentation model
├── train_instance.py            # Trains the instance segmentation model
├── train_vit.py                 # Trains the Vision Transformer model
├── train_gan.py                 # Trains a GAN for cell image synthesis
├── train_cgan.py                # Trains a conditional GAN
├── train_pix2pix.py             # Trains a pix2pix image-to-image model
├── predict.py                   # Runs inference with the segmentation model
├── predict_instance.py          # Runs inference with the instance model
├── track_cells.py               # Tracks cells across frames/sequences
├── compile_video.py             # Compiles frames/predictions into a video
├── show.py                      # Visualization utilities
└── test.py                      # Test/evaluation script
```

## Requirements

- Python 3.x
- PyTorch
- OpenCV (`opencv-python`)
- NumPy

Install dependencies (adjust as needed for your environment, e.g. GPU-enabled PyTorch):

```bash
pip install torch torchvision opencv-python numpy
```

## Usage

1. **Generate synthetic data**

   ```bash
   python run_data_factory.py
   # or generate a specific dataset directly, e.g.
   python generate_cell_data.py
   python generate_instance_data.py
   ```

2. **Train a model**

   ```bash
   python train_segmentation.py     # semantic segmentation (Attention U-Net)
   python train_instance.py         # instance segmentation
   python train_vit.py              # Vision Transformer
   python train_gan.py              # GAN
   python train_cgan.py             # conditional GAN
   python train_pix2pix.py          # pix2pix
   ```

3. **Run inference**

   ```bash
   python predict.py
   python predict_instance.py
   ```

4. **Track cells and compile a video**

   ```bash
   python track_cells.py
   python compile_video.py
   ```

5. **Visualize or evaluate**

   ```bash
   python show.py
   python test.py
   ```

## Notes

- All data used for training is procedurally generated (no real microscopy dataset is bundled), which makes the pipeline easy to run end-to-end without external downloads.
- Script names double as their primary entry points — each can be run directly with `python <script>.py`.
