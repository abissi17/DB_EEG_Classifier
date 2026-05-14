import numpy as np
import torch
import torch.nn as nn

# Jerry's cnn class
class ShuffleAttention_Jerry(nn.Module):
    def __init__(self, channels=128, groups=8):
        super().__init__()

        assert channels % (2 * groups) == 0, \
            "channels must be divisible by 2 * groups"

        self.channels = channels
        self.groups = groups

        branch_channels = channels // (2 * groups)

        # Channel attention branch
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_fc = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Spatial attention branch
        self.spatial_norm = nn.GroupNorm(
            num_groups=1,
            num_channels=branch_channels
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # PyTorch built-in ChannelShuffle
        self.channel_shuffle = nn.ChannelShuffle(groups)

    def forward(self, x):
        B, C, H, W = x.shape
        G = self.groups

        # split feature map into G groups
        x = x.reshape(B * G, C // G, H, W)

        # split each group into channel-attention branch and spatial-attention branch
        x_channel, x_spatial = torch.chunk(x, chunks=2, dim=1)

        # channel attention
        channel_weight = self.channel_fc(self.avg_pool(x_channel))
        x_channel = x_channel * channel_weight

        # spatial attention
        spatial_weight = self.spatial_conv(self.spatial_norm(x_spatial))
        x_spatial = x_spatial * spatial_weight

        # concat two branches
        out = torch.cat([x_channel, x_spatial], dim=1)

        # restore to original shape
        out = out.reshape(B, C, H, W)

        # shuffle channels
        out = self.channel_shuffle(out)

        return out
class CSANet_Jerry(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.6)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.6)
        )

        self.attention = ShuffleAttention_Jerry(channels=128, groups=8)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # (B, 19, 23, 118) -> (B, 1, 19*23, 118)
        x = x.reshape(B, 1, C * H, W)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.attention(x)

        x = self.global_pool(x)
        x = x.flatten(1)

        x = self.fc(x)

        return x

# Jake's cnn class
class ShuffleAttention(nn.Module):
    def __init__(self, channels=128, groups=8):
        super().__init__()
        assert channels % (2 * groups) == 0, "channels must be divisible by 2 * groups"
        self.channels = channels
        self.groups = groups
        branch_channels = channels // (2 * groups)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.spatial_norm = nn.GroupNorm(num_groups=1, num_channels=branch_channels)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.channel_shuffle = nn.ChannelShuffle(groups)

    def forward(self, x):
        B, C, H, W = x.shape
        G = self.groups
        x = x.reshape(B * G, C // G, H, W)
        x_channel, x_spatial = torch.chunk(x, 2, dim=1)

        channel_weight = self.channel_fc(self.avg_pool(x_channel))
        x_channel = x_channel * channel_weight

        spatial_weight = self.spatial_conv(self.spatial_norm(x_spatial))
        x_spatial = x_spatial * spatial_weight

        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.reshape(B, C, H, W)
        out = self.channel_shuffle(out)
        return out
class CSANet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.50)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.60)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.60),
            nn.MaxPool2d(2)
        )
        self.attention = ShuffleAttention(channels=128, groups=8)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, 1, C * H, W)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.attention(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x
