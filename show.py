import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 1. Simulate the Attention Weights we generated in the last step
# Shape: [Batch, Num_Tokens, Num_Tokens] -> [1, 16, 16]
# (In a real run, you would use the 'weights' output from the SelfAttention class)
simulated_weights = torch.rand(1, 16, 16) 

# Let's see what Token 5 is paying attention to
token_index = 5 
# Extract the 16 attention scores for Token 5
token_attention = simulated_weights[0, token_index, :].detach().numpy() 

# 2. Reshape the 16 scores back into a 4x4 spatial grid
attention_grid = token_attention.reshape(4, 4)

# 3. Resize the 4x4 grid back to 64x64 to match our original image
heatmap = cv2.resize(attention_grid, (64, 64), interpolation=cv2.INTER_NEAREST)

# Normalize the heatmap for visualization (0 to 1)
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

# 4. Load the original image to overlay
image_path = "synthetic_dataset/images/synth_cell_0001.png"
original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# 5. Plotting the visualization
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

axes[0].imshow(original_img, cmap="gray")
axes[0].set_title("Original Cell")
axes[0].axis("off")

# Show the raw 4x4 attention map
axes[1].imshow(attention_grid, cmap="viridis")
axes[1].set_title(f"Raw Attention (Token {token_index})")
axes[1].axis("off")

# Overlay the heatmap on the image
axes[2].imshow(original_img, cmap="gray")
axes[2].imshow(heatmap, cmap="jet", alpha=0.5) # alpha adds transparency
axes[2].set_title(f"Attention Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("attention_heatmap.png", bbox_inches='tight')
print("Attention heatmap successfully rendered and saved to disk!")