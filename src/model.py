"""
3D ResNet models for Harvard-GF OCT TDS regression.

Clean version:
  - baseline model: forward(x)
  - fairness-aware model: forward(x, attr)
  - demographic embedding retained
  - subgroup-specific calibration layers retained for AFF
  - group_specific_layers exposed as nn.ModuleList for optimizer param groups
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()

        self.conv1 = nn.Conv3d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Identity()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(planes),
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = F.relu(out, inplace=True)

        return out


class ChannelAttention3D(nn.Module):
    """
    Lightweight channel attention after global pooling.
    Input and output shape: [B, C]
    """

    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 16)

        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        weights = self.net(x)
        return x * weights


class ResNet3DBackbone(nn.Module):
    """
    Compact 3D ResNet-18-style backbone for 200x200x200 OCT volumes.

    Input:
        x: [B, 1, D, H, W]

    Output:
        feat: [B, feature_dim]
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=32,
        feature_dim=512,
    ):
        super().__init__()

        self.in_planes = base_channels

        # Aggressive early downsampling is important for 200^3 volumes.
        self.stem = nn.Sequential(
            nn.Conv3d(
                in_channels,
                base_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(base_channels, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(base_channels * 2, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(base_channels * 4, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(base_channels * 8, num_blocks=2, stride=2)

        backbone_out_dim = base_channels * 8

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.attn = ChannelAttention3D(backbone_out_dim)

        self.proj = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        self.feature_dim = feature_dim

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)

        layers = []
        for s in strides:
            layers.append(BasicBlock3D(self.in_planes, planes, stride=s))
            self.in_planes = planes * BasicBlock3D.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)

        x = self.attn(x)
        x = self.proj(x)

        return x


class ResNet3D_Baseline(nn.Module):
    """
    Baseline 3D ResNet for 52-point TDS regression.

    Severity removed.
    Demographic attributes are not used.

    Forward:
        pred = model(x)
    """

    def __init__(
        self,
        in_channels=1,
        out_dim=52,
        feature_dim=512,
        base_channels=32,
        pretrained_backbone=False,
        **kwargs,
    ):
        super().__init__()

        # pretrained_backbone kept only for compatibility with older scripts.
        # It is not used here because this is a custom OCT 3D model.
        self.backbone = ResNet3DBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            feature_dim=feature_dim,
        )

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.regressor(feat)


class GroupSpecificCalibration(nn.Module):
    """
    Group-specific feature calibration.

    This is intentionally lightweight and is the module family updated by AFF.
    """

    def __init__(self, feature_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

        # Start close to identity behavior.
        self.scale = nn.Parameter(torch.ones(feature_dim))
        self.bias = nn.Parameter(torch.zeros(feature_dim))

    def forward(self, feat):
        delta = self.net(feat)
        calibrated = feat + 0.1 * delta
        calibrated = calibrated * self.scale + self.bias
        return calibrated


class FairResNet3D_R18_Attn(nn.Module):
    """
    Fairness-aware 3D ResNet with demographic embedding and group-specific calibration.

    Severity removed.

    Forward:
        pred = model(x, attr)

    Args:
        x:    [B, 1, D, H, W]
        attr: [B], integer group labels
              race:     0 Asian, 1 Black, 2 White
              hispanic: 0 Non-Hispanic, 1 Hispanic

    Important:
        self.group_specific_layers is exposed as nn.ModuleList so the training
        script can create optimizer parameter groups and apply AFF updates only
        to subgroup-specific calibration layers.
    """

    def __init__(
        self,
        in_channels=1,
        out_dim=52,
        num_groups=3,
        attr_emb_dim=128,
        feature_dim=512,
        base_channels=32,
        pretrained_backbone=False,
        **kwargs,
    ):
        super().__init__()

        self.num_groups = num_groups
        self.attr_emb_dim = attr_emb_dim
        self.feature_dim = feature_dim

        self.backbone = ResNet3DBackbone(
            in_channels=in_channels,
            base_channels=base_channels,
            feature_dim=feature_dim,
        )

        self.attr_embedding = nn.Embedding(num_groups, attr_emb_dim)

        self.attr_mlp = nn.Sequential(
            nn.Linear(attr_emb_dim, attr_emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(attr_emb_dim, attr_emb_dim),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim + attr_emb_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

        self.group_specific_layers = nn.ModuleList(
            [GroupSpecificCalibration(feature_dim) for _ in range(num_groups)]
        )

        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, out_dim),
        )

    def forward(self, x, attr):
        attr = attr.long().view(-1)

        if torch.any(attr < 0) or torch.any(attr >= self.num_groups):
            raise ValueError(
                f"attr contains labels outside [0, {self.num_groups - 1}]. "
                f"Got min={attr.min().item()}, max={attr.max().item()}."
            )

        visual_feat = self.backbone(x)

        attr_feat = self.attr_embedding(attr)
        attr_feat = self.attr_mlp(attr_feat)

        fused = torch.cat([visual_feat, attr_feat], dim=1)
        fused = self.fusion(fused)

        # Apply the calibration layer corresponding to each sample's group.
        calibrated = torch.empty_like(fused)

        for gid in range(self.num_groups):
            mask = attr == gid
            if mask.any():
                calibrated[mask] = self.group_specific_layers[gid](fused[mask])

        pred = self.regressor(calibrated)
        return pred
