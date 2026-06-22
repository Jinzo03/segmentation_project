import torch
import torch.nn as nn
import math

# 1. The Attention Engine (What we built earlier)
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3) # Efficiently combine Q,K,V projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, Seq_Len, Embed_Dim = x.shape
        qkv = self.qkv_proj(x).chunk(3, dim=-1) # Split into Q, K, V
        Q, K, V = qkv
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.embed_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        
        return self.out_proj(context)

# 2. The Transformer Block (Attention + Normalization + Feed Forward)
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # The MLP brain that processes the attention insights
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        # Notice the Skip Connections (x + ...), just like your U-Net!
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# 3. The Complete Vision Transformer
class SimpleViT(nn.Module):
    def __init__(self, image_size=64, patch_size=16, in_channels=1, embed_dim=256, num_classes=2):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        
        # Trick: A 2D Convolution with a stride equal to the patch size is the 
        # mathematically identical, highly optimized way to extract and flatten patches!
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Trainable GPS Coordinates
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Stack multiple Transformer Blocks (Let's use 4)
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim) for _ in range(4)])
        
        # Final Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        # 1. Patchify and Flatten: [B, 1, 64, 64] -> [B, 256, 4, 4] -> [B, 16, 256]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        
        # 2. Add GPS map
        x = x + self.pos_embedding
        
        # 3. Pass through the Attention Blocks
        x = self.blocks(x)
        
        # 4. To classify the image, we average all the patches together into one vector
        x = x.mean(dim=1) 
        x = self.norm(x)
        
        # 5. Output prediction
        return self.head(x)

# --- Ignition Sequence ---
model = SimpleViT()
dummy_image = torch.randn(1, 1, 64, 64)
prediction = model(dummy_image)

print(f"ViT Assembled! Output Shape: {prediction.shape} (Batch Size, Num Classes)")