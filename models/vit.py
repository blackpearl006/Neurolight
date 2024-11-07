import torch.nn as nn
from monai.networks.nets import ViT

class ViTWithRegression(nn.Module):
    def __init__(self):
        super(ViTWithRegression, self).__init__()
        self.vit = ViT(
            in_channels=1,
            img_size=(112, 112, 112),  # Input image size
            patch_size=(16, 16, 16),   # Patch size
            classification=False       # Disable classification head
        )
        
        # Global Average Pooling layer to reduce the feature dimension
        self.gap = nn.AdaptiveAvgPool1d(1)  # Pooling over the patch dimension
        
        # Regression head layers
        self.regression_head = nn.Sequential(
            nn.Linear(in_features=768, out_features=256, bias=True),  # Hidden dimension from ViT
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=16, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=16, out_features=1, bias=True)  # Regression output
        )

    def forward(self, x):
        vit_output = self.vit(x)
        vit_tensor = vit_output[0]
        vit_tensor = vit_tensor.permute(0, 2, 1)
        pooled_output = self.gap(vit_tensor).squeeze(-1)
        out = self.regression_head(pooled_output)
        return out