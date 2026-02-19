import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18


class ChannelAttention(nn.Module):
    """
    Simple channel-wise attention on pooled feature vector.
    """
    def __init__(self, in_dim, reduction=4):
        super().__init__()
        hidden = max(in_dim // reduction, 16)
        self.attn = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C]
        weights = self.attn(x)
        return x * weights


class FairResNet3D_R18_Attn(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_dim=52,
        num_groups=3,
        attr_emb_dim=128,
        pretrained_backbone=False
    ):
        super().__init__()
        backbone = r3d_18(pretrained=pretrained_backbone)
        backbone.stem[0] = nn.Conv3d(
            in_channels,
            64,
            kernel_size=(3, 7, 7),
            stride=(1, 2, 2),
            padding=(1, 3, 3),
            bias=False
        )

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # remove classifier
        self.feature_dim = backbone.fc.in_features  
        self.attention = ChannelAttention(self.feature_dim)

       
        self.attr_embedding = nn.Sequential(
            nn.Embedding(num_groups, attr_emb_dim),
            nn.Linear(attr_emb_dim, attr_emb_dim),
            nn.ReLU(inplace=True)
        )

       
        self.group_specific_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim + attr_emb_dim, self.feature_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_groups)
        ])

       
        self.head = nn.Linear(self.feature_dim, out_dim)

    def forward(self, x, group_ids):
        """
        x: [B, 1, D, H, W]
        group_ids: [B]
        """

        # Backbone feature extraction
        features = self.backbone(x)  
        features = features.flatten(1) 

        # Channel attention
        features = self.attention(features)

        # Attribute embedding
        attr_emb = self.attr_embedding[0](group_ids.long())
        attr_emb = self.attr_embedding[1:](attr_emb)

        # Concatenate features + attribute embedding
        fused = torch.cat([features, attr_emb], dim=1)

        # Apply group-specific calibration
        calibrated = torch.zeros_like(features)
        for gid in range(len(self.group_specific_layers)):
            mask = (group_ids == gid)
            if mask.sum() > 0:
                calibrated[mask] = self.group_specific_layers[gid](fused[mask])

        # Final regression
        output = self.head(calibrated)

        return output