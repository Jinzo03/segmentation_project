import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    [Standard]
    Two consecutive 3x3 convolutions, each followed by BatchNorm and ReLU.
    This remains identical to your previous version.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class AttentionGate(nn.Module):
    """
    [New to Attention U-Net]
    Filters the skip connection (x) using the decoder feature (g) as a gating signal.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        # F_g: channels of the gating signal (from lower decoder layer)
        # F_l: channels of the skip connection (from encoder layer)
        # F_int: intermediate channel size (usually F_g // 2)

        # W_g: Process the gating signal g
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # W_x: Process the skip connection x
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        # psi: Final attention coefficient generation
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid() # Limits coefficients between 0 (background) and 1 (object)
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        # 1. Process inputs through W_g and W_x
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        # 2. Sum them (they must have the same spatial size and intermediate channels)
        # We need to make sure the spatial sizes match. 
        # Usually g (the upsampled signal) has the correct size.
        if g1.size()[2:] != x1.size()[2:]:
             g1 = F.interpolate(g1, size=x1.size()[2:], mode='bilinear', align_corners=True)

        # 3. Apply ReLU activation
        psi = self.relu(g1 + x1)
        
        # 4. Apply psi to get coefficients and sigmoid
        psi = self.psi(psi)
        
        # 5. Multiply original skip connection x by the attention mask
        return x * psi

class AttentionUNet(nn.Module):
    """
    [Upgraded]
    U-Net structure modified to include Attention Gates before concatenation in the decoder.
    """
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(AttentionUNet, self).__init__()
        
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder (Downsampling)
        # Stays identical to standard U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature
            
        # Bottleneck (Lowest resolution)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Decoder (Upsampling + Attention + Merging)
        self.ups = nn.ModuleList()
        self.atts = nn.ModuleList() # List for AttentionGates
        
        # Loop backwards through features
        for feature in reversed(features):
            # 1. Define upsampling operation (e.g., ConvTranspose or Upsample)
            # We use Upsample here to keep things simple with the AG integration.
            self.ups.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(feature*2, feature, kernel_size=1) # Reduce channels
                )
            )
            
            # 2. NEW: Define an Attention Gate for this merge step
            # AG receives gating (feature from bottleneck/lower decoder) and skip (feature from encoder)
            # Intermediate channels = feature // 2
            self.atts.append(AttentionGate(F_g=feature, F_l=feature, F_int=feature // 2))
            
            # 3. Final processing DoubleConv block after merging
            # Input channels = feature (upsampled signal) + feature (attentive skip connection)
            self.ups.append(DoubleConv(feature*2, feature))
            
        # Final Output Layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            
        # Bottleneck
        x = self.bottleneck(x)
        
        skip_connections = skip_connections[::-1] # Reverse for decoding loop
        
        # Decoder (Using index stepping: 0=up/reduction, 1=AG, 2=FinalConv)
        # This list structure can be confusing. Let's make it explicit:
        for idx in range(0, len(self.ups), 2):
            up_layer = self.ups[idx]
            final_double_conv = self.ups[idx+1]
            attention_gate = self.atts[idx//2] # Map upsampling index to AG index
            
            # 1. Upsample gating signal (g)
            x = up_layer(x) 
            
            # 2. Get matching skip connection (x_skip)
            skip_connection = skip_connections[idx//2]
            
            # 3. NEW: Filter the skip connection via the Attention Gate
            attentive_skip = attention_gate(skip_connection, x)
            
            # 4. Concatenate upsampled signal and attentive skip connection
            x = torch.cat((x, attentive_skip), dim=1)
            
            # 5. Final processing
            x = final_double_conv(x)
            
        return self.final_conv(x)

def test_compile():
    """
    Sanity check to verify shapes.
    """
    model = AttentionUNet(in_channels=3, out_channels=3)
    x = torch.randn((1, 3, 256, 256))
    preds = model(x)
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {preds.shape}")
    assert preds.shape == (1, 3, 256, 256)
    print("AttentionUnet compiled successfully.")

if __name__ == "__main__":
    test_compile()