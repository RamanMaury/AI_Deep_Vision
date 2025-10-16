# models/csrnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CSRNet(nn.Module):
    """
    Front-end: VGG16 (first 23 layers) -> keeps reasonable spatial size
    Back-end: dilated convs -> 1-channel density map
    Output is upsampled back to input HxW for a simple MSE loss with GT.
    """
    def __init__(self, upsample_to_input=True):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:23])  # up to relu4_3

        # Dilated conv backend (common CSRNet style)
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(inplace=True),
        )

        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        self.upsample_to_input = upsample_to_input

    def forward(self, x):
        h, w = x.shape[2], x.shape[3]
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)   # [B,1,H/8,W/8]-ish
        if self.upsample_to_input:
            x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
        return x
 
if __name__ == "__main__":
    model = CSRNet() 
    dummy = torch.randn(1, 3, 576, 768)   # fake input image [B,C,H,W]
    output = model(dummy)
    print("✅ Model built successfully")
    print("Input shape :", tuple(dummy.shape))
    print("Output shape:", tuple(output.shape))
 