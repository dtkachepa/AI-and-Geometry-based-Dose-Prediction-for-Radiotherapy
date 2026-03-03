"""
RANDose + Mamba SSM Integration
================================
Implements the two integration strategies from the proposal:

    Strategy A (Sequential, eq. 11):
        F_out = CSA(Mamba(MSFE(F_in)))

    Strategy B (Parallel, eq. 12):
        F_out = CSA(MSFE(F_in)) + γ · Mamba(MSFE(F_in))

Memory note — Mamba is enabled only at stages 3-5 (spatial dims ≤ 32³).
At stages 1-2 (128³ and 64³) the TriDirectionalMamba reshape produces
batch sizes of 32 768 and 8 192 respectively, which OOMs even on an H100.
Stages 3-5 produce (2 048, 32, C), (512, 16, C), (128, 8, C) — all safe.
Early stages fall back to pure MSFE + CSA (RANDose baseline behaviour).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.se3d import ChannelSpatialSELayer3D
from utils.mamba_3d import TriDirectionalMamba


# ── Shared building blocks ──────────────────────────────────────────────────

class UpConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=1, bias=True),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=True)
        return self.conv(x)


class Comb(nn.Module):
    """PI module: FiLM-style PTV conditioning — out = enc * (1 + γ) + β."""
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.gamma_layer = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.beta_layer = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.channel_expand = nn.Conv3d(1, in_channels, kernel_size=1)

    def forward(self, enc_out, ptv):
        ptv_r = F.interpolate(ptv, size=enc_out.shape[2:], mode='trilinear', align_corners=False)
        if ptv_r.shape[1] != self.in_channels:
            ptv_r = self.channel_expand(ptv_r)
        gamma = self.gamma_layer(ptv_r)
        beta = self.beta_layer(ptv_r)
        return enc_out * (1 + gamma) + beta


class AddFeatureMaps(nn.Module):
    """AF module: attention-weighted skip connection fusion."""
    def __init__(self, in_channels):
        super().__init__()
        self.fusion_weight1 = nn.Parameter(torch.ones(1))
        self.fusion_weight2 = nn.Parameter(torch.ones(1))
        self.attention = ChannelSpatialSELayer3D(in_channels)

    def forward(self, F1, F2):
        return self.fusion_weight1 * self.attention(F1) + self.fusion_weight2 * self.attention(F2)


# ── Strategy A: Sequential MSCSA + Mamba ───────────────────────────────────

class MSCSAMambaBlockA(nn.Module):
    """
    Strategy A — Sequential integration (eq. 11):
        F_out = CSA(Mamba(MSFE(F_in))) + shortcut   [when use_mamba=True]
        F_out = CSA(MSFE(F_in))        + shortcut   [when use_mamba=False]

    use_mamba=False is used for stages 1-2 to avoid OOM at full resolution.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 d_state: int = 16, use_mamba: bool = True):
        super().__init__()
        self.use_mamba = use_mamba

        # MSFE: parallel 3D convolutions with kernels {3, 5, 7, 9}
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(in_ch, out_ch, kernel_size=5, stride=stride, padding=2)
        self.conv3 = nn.Conv3d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3)
        self.conv4 = nn.Conv3d(in_ch, out_ch, kernel_size=9, stride=stride, padding=4)
        self.nonlin = nn.LeakyReLU(inplace=True)

        if use_mamba:
            # Mamba: tri-directional scanning (axial / coronal / sagittal)
            self.mamba = TriDirectionalMamba(out_ch, d_state=d_state)

        # CSA: concurrent channel-spatial squeeze-and-excitation
        self.csa = ChannelSpatialSELayer3D(out_ch, reduction_ratio=2)

        # Residual shortcut (only when channels or spatial dims change)
        self.shortcut = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride)
            if in_ch != out_ch or stride != 1 else None
        )

    def forward(self, x):
        # MSFE
        msfe = (self.nonlin(self.conv1(x))
                + self.nonlin(self.conv2(x))
                + self.nonlin(self.conv3(x))
                + self.nonlin(self.conv4(x)))

        # Mamba (sequential: between MSFE and CSA)
        if self.use_mamba:
            msfe = self.mamba(msfe)

        # CSA
        out = self.csa(msfe)

        # Residual shortcut
        if self.shortcut is not None:
            out = out + self.shortcut(x)

        return out


# ── Strategy B: Parallel MSCSA + Mamba ─────────────────────────────────────

class MSCSAMambaBlockB(nn.Module):
    """
    Strategy B — Parallel integration (eq. 12):
        F_out = CSA(MSFE(F_in)) + γ·Mamba(MSFE(F_in)) + shortcut  [use_mamba=True]
        F_out = CSA(MSFE(F_in))                        + shortcut  [use_mamba=False]
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int,
                 d_state: int = 16, use_mamba: bool = True):
        super().__init__()
        self.use_mamba = use_mamba

        # MSFE: parallel 3D convolutions with kernels {3, 5, 7, 9}
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv3d(in_ch, out_ch, kernel_size=5, stride=stride, padding=2)
        self.conv3 = nn.Conv3d(in_ch, out_ch, kernel_size=7, stride=stride, padding=3)
        self.conv4 = nn.Conv3d(in_ch, out_ch, kernel_size=9, stride=stride, padding=4)
        self.nonlin = nn.LeakyReLU(inplace=True)

        # CSA branch (always active)
        self.csa = ChannelSpatialSELayer3D(out_ch, reduction_ratio=2)

        if use_mamba:
            # Mamba parallel branch + learnable fusion weight γ
            self.mamba = TriDirectionalMamba(out_ch, d_state=d_state)
            self.gamma = nn.Parameter(torch.ones(1))

        # Residual shortcut
        self.shortcut = (
            nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride)
            if in_ch != out_ch or stride != 1 else None
        )

    def forward(self, x):
        # MSFE
        msfe = (self.nonlin(self.conv1(x))
                + self.nonlin(self.conv2(x))
                + self.nonlin(self.conv3(x))
                + self.nonlin(self.conv4(x)))

        # Parallel branches
        out = self.csa(msfe)
        if self.use_mamba:
            out = out + self.gamma * self.mamba(msfe)

        # Residual shortcut
        if self.shortcut is not None:
            out = out + self.shortcut(x)

        return out


# ── Encoder ─────────────────────────────────────────────────────────────────
#
# Mamba stage assignment (spatial dims after block, with 128³ input):
#   enc1: 128³  B·H·W = 32 768  → use_mamba=False  (OOM risk)
#   enc2:  64³  B·H·W =  8 192  → use_mamba=False  (OOM risk)
#   enc3:  32³  B·H·W =  2 048  → use_mamba=True   ✓
#   enc4:  16³  B·H·W =    512  → use_mamba=True   ✓
#   enc5:   8³  B·H·W =    128  → use_mamba=True   ✓

class Encoder(nn.Module):
    def __init__(self, block_cls, in_ch, list_ch, d_state):
        super().__init__()
        self.enc1 = block_cls(in_ch,      list_ch[1], stride=1, d_state=d_state, use_mamba=False)
        self.enc2 = block_cls(list_ch[1], list_ch[2], stride=2, d_state=d_state, use_mamba=False)
        self.enc3 = block_cls(list_ch[2], list_ch[3], stride=2, d_state=d_state, use_mamba=True)
        self.enc4 = block_cls(list_ch[3], list_ch[4], stride=2, d_state=d_state, use_mamba=True)
        self.enc5 = block_cls(list_ch[4], list_ch[5], stride=2, d_state=d_state, use_mamba=True)

        # PTV integration (PI) for stages 2–5
        self.comb2 = Comb(list_ch[1])
        self.comb3 = Comb(list_ch[2])
        self.comb4 = Comb(list_ch[3])
        self.comb5 = Comb(list_ch[4])

    def forward(self, x):
        ptv = x[:, 0:1]
        e1 = self.enc1(x)
        e2 = self.enc2(self.comb2(e1, ptv))
        e3 = self.enc3(self.comb3(e2, ptv))
        e4 = self.enc4(self.comb4(e3, ptv))
        e5 = self.enc5(self.comb5(e4, ptv))
        return [e1, e2, e3, e4, e5]


# ── Decoder ─────────────────────────────────────────────────────────────────
#
# Mamba stage assignment (spatial dims of the decoder feature map):
#   dec4:  16³  → use_mamba=True   ✓
#   dec3:  32³  → use_mamba=True   ✓
#   dec2:  64³  → use_mamba=False  (OOM risk)
#   dec1: 128³  → use_mamba=False  (OOM risk)

class Decoder(nn.Module):
    def __init__(self, block_cls, list_ch, d_state):
        super().__init__()
        self.up4 = UpConv(list_ch[5], list_ch[4])
        self.up3 = UpConv(list_ch[4], list_ch[3])
        self.up2 = UpConv(list_ch[3], list_ch[2])
        self.up1 = UpConv(list_ch[2], list_ch[1])

        self.dec4 = block_cls(list_ch[4], list_ch[4], stride=1, d_state=d_state, use_mamba=True)
        self.dec3 = block_cls(list_ch[3], list_ch[3], stride=1, d_state=d_state, use_mamba=True)
        self.dec2 = block_cls(list_ch[2], list_ch[2], stride=1, d_state=d_state, use_mamba=False)
        self.dec1 = block_cls(list_ch[1], list_ch[1], stride=1, d_state=d_state, use_mamba=False)

        # AF modules: attention-weighted skip connection fusion
        self.fuse4 = AddFeatureMaps(list_ch[4])
        self.fuse3 = AddFeatureMaps(list_ch[3])
        self.fuse2 = AddFeatureMaps(list_ch[2])
        self.fuse1 = AddFeatureMaps(list_ch[1])

    def forward(self, enc_feats):
        e1, e2, e3, e4, e5 = enc_feats
        d4 = self.dec4(self.fuse4(self.up4(e5), e4))
        d3 = self.dec3(self.fuse3(self.up3(d4), e3))
        d2 = self.dec2(self.fuse2(self.up2(d3), e2))
        d1 = self.dec1(self.fuse1(self.up1(d2), e1))
        return d1


# ── Base U-Net ──────────────────────────────────────────────────────────────

class BaseUNet(nn.Module):
    def __init__(self, block_cls, in_ch, list_ch, d_state):
        super().__init__()
        self.encoder = Encoder(block_cls, in_ch, list_ch, d_state)
        self.decoder = Decoder(block_cls, list_ch, d_state)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ── Top-level model classes ─────────────────────────────────────────────────

class Model_RANDose_MambaA(nn.Module):
    """
    RANDose + Strategy A Mamba integration (Sequential, eq. 11).
    Mamba applied at encoder stages 3-5 and decoder stages 4-3.
    """
    def __init__(self, in_ch, out_ch, list_ch_A, list_ch_B,
                 d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        self.net = BaseUNet(MSCSAMambaBlockA, in_ch, list_ch_A, d_state)
        self.conv_out = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        return self.conv_out(self.net(x))


class Model_RANDose_MambaB(nn.Module):
    """
    RANDose + Strategy B Mamba integration (Parallel, eq. 12).
    Mamba applied at encoder stages 3-5 and decoder stages 4-3.
    """
    def __init__(self, in_ch, out_ch, list_ch_A, list_ch_B,
                 d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        self.net = BaseUNet(MSCSAMambaBlockB, in_ch, list_ch_A, d_state)
        self.conv_out = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1, bias=True)

    def forward(self, x):
        return self.conv_out(self.net(x))


if __name__ == '__main__':
    import numpy as np
    for name, cls in [('MambaA', Model_RANDose_MambaA),
                      ('MambaB', Model_RANDose_MambaB)]:
        model = cls(in_ch=9, out_ch=1,
                    list_ch_A=[-1, 16, 32, 64, 128, 256],
                    list_ch_B=[-1, 32, 64, 128, 256, 512],
                    d_state=16, d_conv=4, expand=2)
        params = np.sum([p.numel() for p in model.parameters()]) * 4e-6
        print(f'{name}: {params:.2f} MB params')
        out = model(torch.randn(1, 9, 128, 128, 128))
        print(f'{name}: output {out.shape}')
