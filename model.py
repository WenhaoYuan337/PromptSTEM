import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class CBAM(nn.Module):
    """通道注意力 + 空间注意力"""

    def __init__(self, channels):
        super(CBAM, self).__init__()
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.channel_att(x) * x
        spatial = torch.cat([att.max(dim=1, keepdim=True)[0], att.mean(dim=1, keepdim=True)], dim=1)
        att = self.spatial_att(spatial) * att
        return att


class MSFA(nn.Module):
    """多尺度特征增强"""

    def __init__(self, in_channels, out_channels):
        super(MSFA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        self.conv_out = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return self.conv_out(x)


class UNet(nn.Module):
    def __init__(self, encoder_name="resnet34", msfa=True, cbam=True):
        super(UNet, self).__init__()
        resnet = resnet34(pretrained=True)
        self.encoder_layers = list(resnet.children())[:6]  # 保留前几层作为编码器
        self.encoder = nn.Sequential(*self.encoder_layers)

        self.msfa = MSFA(128, 128) if msfa else nn.Identity()
        self.cbam = CBAM(128) if cbam else nn.Identity()

        self.upsample = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.msfa(x)
        x = self.cbam(x)
        x = self.upsample(x)
        x = self.final_conv(x)
        x = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)  # **添加上采样**
        return x

class SimCLR(nn.Module):
    def __init__(self, base_model):
        super(SimCLR, self).__init__()
        self.encoder = base_model.encoder  # 共享 UNet 编码器

        # **计算编码器的输出维度**
        sample_input = torch.randn(1, 3, 256, 256)  # 假设输入
        sample_output = self.encoder(sample_input)
        feature_dim = sample_output.shape[1] * sample_output.shape[2] * sample_output.shape[3]

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512),  # **自动适配 feature_dim**
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  # 展平
        x = self.projection(x)
        return x
