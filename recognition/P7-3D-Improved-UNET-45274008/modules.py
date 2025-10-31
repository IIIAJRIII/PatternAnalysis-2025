import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Building Blocks ---

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # increase spatial resolution while decreasing feature resolution
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4] # 3D difference

        x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
                        diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# --- Full 3D UNet Model ---

class UNet3D(nn.Module):
    # Updated to accept base_features, defaulting to 32
    def __init__(self, n_channels, n_classes, base_features=32, bilinear=False):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder path
        self.inc = DoubleConv(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)
        factor = 2 if bilinear else 1
        # The bottleneck layer
        self.down4 = Down(base_features * 8, (base_features * 16) // factor) 
        
        # Decoder path
        self.up1 = Up(base_features * 16, (base_features * 8) // factor, bilinear)
        self.up2 = Up(base_features * 8, (base_features * 4) // factor, bilinear)
        self.up3 = Up(base_features * 4, (base_features * 2) // factor, bilinear)
        self.up4 = Up(base_features * 2, base_features, bilinear) # No factor on last 'out_channels'
        self.outc = OutConv(base_features, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# ----------------------------------------------------------------------
# --- Building Blocks for Improved 3D UNet ---
# --- Uses InstanceNorm, LeakyReLU, and Residual Connections ---
# ----------------------------------------------------------------------

class ImprovedConvBlock(nn.Module):
    """
    (convolution => [IN] => LeakyReLU) * 2 + [Residual Shortcut]
    As described in the "Improved 3D UNet" paper.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm3d(out_channels)
        
        # Shortcut connection to match dimensions
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            # 1x1 conv to match channel dimension
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.InstanceNorm3d(out_channels)
            )
            
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.lrelu(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out += residual
        out = self.lrelu(out) # Final activation after addition
        return out


class ImprovedDown(nn.Module):
    """Downscaling with maxpool then ImprovedConvBlock"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ImprovedConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ImprovedUp(nn.Module):
    """Upscaling then ImprovedConvBlock"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        """
        'in_channels' is the TOTAL number of channels after concatenation
        'out_channels' is the number of output channels
        """
        super().__init__()
        self.out_channels = out_channels

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = ImprovedConvBlock(in_channels, out_channels)
        else:
            # The 'in_channels' for this block is the concatenated size (e.g., 256 + 128 = 384)
            # The 'out_channels' is the skip connection size (e.g., 128)
            # Therefore, the channels from the layer below (x1) must be in_channels - out_channels
            x1_channels = in_channels - out_channels # e.g., 384 - 128 = 256
            
            # The ConvTranspose3d must take x1_channels and output out_channels
            # to match the skip connection (x2)
            self.up = nn.ConvTranspose3d(x1_channels, out_channels, kernel_size=2, stride=2)
            self.conv = ImprovedConvBlock(out_channels * 2, out_channels)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        diffZ = x2.size()[4] - x1.size()[4] # 3D difference

        x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
                       diffX // 2, diffX - diffX // 2,
                       diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# --- Full Improved 3D UNet Model (Hard Difficulty) ---

class ImprovedUNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, base_features=32, bilinear=False):
        super(ImprovedUNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = ImprovedConvBlock(n_channels, base_features)
        self.down1 = ImprovedDown(base_features, base_features * 2)
        self.down2 = ImprovedDown(base_features * 2, base_features * 4)
        self.down3 = ImprovedDown(base_features * 4, base_features * 8)
        self.down4 = ImprovedDown(base_features * 8, base_features * 16) 
        
        # Decoder path
        # The 'in_channels' for ImprovedUp is the total from (x1_upsampled + x2_skip)
        # The 'out_channels' is the size of the skip connection (x2_skip)
        self.up1 = ImprovedUp(base_features * 16 + base_features * 8, base_features * 8, bilinear)
        self.up2 = ImprovedUp(base_features * 8 + base_features * 4, base_features * 4, bilinear)
        self.up3 = ImprovedUp(base_features * 4 + base_features * 2, base_features * 2, bilinear)
        self.up4 = ImprovedUp(base_features * 2 + base_features, base_features, bilinear)
        self.outc = OutConv(base_features, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits