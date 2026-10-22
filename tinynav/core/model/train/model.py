"""BevNet: infra1 image features sampled into a BEV volume (Simple-BEV style) + runtime-depth raster -> BEV geometry.

Inputs  img (B,1,H,W) in [0,1], raster (B,4,N,N) from depth_raster, grid (B,Z*N,N,2) and valid (B,Z,N,N)
        from sampling_grid. Grids are computed outside the network so the export only needs grid_sample.
Outputs (B,C,N,N) logits: known, ground (m, rel. camera; regression), span >= SPAN_T[i], step >= STEP_T[i], rail.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bev_geometry import BEV_N, Z_SLICES
from .cost import SPAN_T, STEP_T

# name -> channels
HEADS = dict(known=1, ground=1, span=len(SPAN_T), step=len(STEP_T), rail=1)


def conv_bn(cin, cout, k=3, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ImageEncoder(nn.Module):
    """ResNet-18 features at strides 8 and 16, merged at stride 8."""

    def __init__(self, out_ch=64, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('resnet18', pretrained=pretrained, features_only=True,
                                          out_indices=(2, 3), in_chans=1)
        c8, c16 = self.backbone.feature_info.channels()
        self.lat8 = nn.Conv2d(c8, out_ch, 1)
        self.lat16 = nn.Conv2d(c16, out_ch, 1)
        self.out = conv_bn(out_ch, out_ch)

    def forward(self, x):
        f8, f16 = self.backbone((x - 0.45) / 0.25)
        f = self.lat8(f8) + F.interpolate(self.lat16(f16), size=f8.shape[-2:], mode='bilinear', align_corners=False)
        return self.out(f)


class BevUNet(nn.Module):
    def __init__(self, cin, c=(64, 128, 256)):
        super().__init__()
        self.e1 = nn.Sequential(conv_bn(cin, c[0]), conv_bn(c[0], c[0]))
        self.e2 = nn.Sequential(conv_bn(c[0], c[1], s=2), conv_bn(c[1], c[1]))
        self.e3 = nn.Sequential(conv_bn(c[1], c[2], s=2), conv_bn(c[2], c[2]), conv_bn(c[2], c[2]))
        self.d2 = nn.Sequential(conv_bn(c[2] + c[1], c[1]), conv_bn(c[1], c[1]))
        self.d1 = nn.Sequential(conv_bn(c[1] + c[0], c[0]), conv_bn(c[0], c[0]))

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(x1)
        x3 = self.e3(x2)
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode='bilinear', align_corners=False)
        y2 = self.d2(torch.cat([up(x3, x2), x2], 1))
        return self.d1(torch.cat([up(y2, x1), x1], 1))


class BevNet(nn.Module):
    def __init__(self, feat_ch=64, pretrained=True):
        super().__init__()
        nz = len(Z_SLICES)
        self.encoder = ImageEncoder(feat_ch, pretrained)
        self.squeeze = conv_bn(feat_ch * nz + nz, 96, k=1)
        self.unet = BevUNet(96 + 4)
        self.head = nn.Conv2d(64, sum(HEADS.values()), 1)

    def forward(self, img, raster, grid, valid):
        feat = self.encoder(img)
        b, c = feat.shape[:2]
        nz = valid.shape[1]
        vol = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        vol = vol.view(b, c * nz, BEV_N, BEV_N)
        bev = self.squeeze(torch.cat([vol, valid], 1))
        return self.head(self.unet(torch.cat([bev, raster], 1)))


def split_heads(out):
    """Single-channel heads come back as (B,N,N), threshold heads as (B,T,N,N)."""
    res, i = {}, 0
    for k, n in HEADS.items():
        res[k] = out[:, i] if n == 1 else out[:, i:i + n]
        i += n
    return res
