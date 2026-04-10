"""
Model_RANDose_MambaVision
=========================
Adapts the MambaVision hybrid design (Hatamizadeh & Kautz, CVPR 2025) to 3D
dose prediction inside the RANDose U-Net framework.

Key ideas from MambaVision applied here:
  1. MambaVision3DMixer  — replaces causal conv with regular 3D conv; adds a
     symmetric non-SSM branch (Conv3D + SiLU); concat both branches.
     This gives the big accuracy jump shown in Table 4 of the paper.
  2. Hybrid encoder blocks — first 3 encoder stages use MambaVision3DMixer
     for efficient local + long-range feature extraction.
  3. Self-attention in the final decoder stages (decoder_1, decoder_2) —
     the paper's key finding: self-attention at the *end* best captures
     global context / long-range spatial dependencies (Table 5, last N/2).
     For dose prediction this matters at full resolution where boundary
     precision counts most.
  4. PI (Comb) and AF (AddFeatureMaps) modules kept identical to Model_MTASP
     so the RANDose region-aware machinery is preserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.se3d import ChannelSpatialSELayer3D


# ─────────────────────────────────────────────────────────────────────────────
# 1. MambaVision 3D Mixer
# ─────────────────────────────────────────────────────────────────────────────

class MambaVision3DMixer(nn.Module):
    """
    3D adaptation of the MambaVision mixer block (eq. 7 in the paper).

    Two parallel branches from a shared linear projection:
        Branch 1 (SSM path):  Linear(C→C/2) → Conv3D → TriDirectionalMamba → SiLU
        Branch 2 (conv path): Linear(C→C/2) → Conv3D → SiLU
    Output: Linear(C→C)(Concat(B1, B2))   — same channel count in/out

    The symmetric branch compensates for information lost due to the sequential
    constraints of the SSM, giving richer feature representations.
    """

    def __init__(self, channels: int, d_state: int = 16):
        super().__init__()
        half = channels // 2

        # Shared input projection splits channels in half for each branch
        self.in_proj = nn.Conv3d(channels, channels, kernel_size=1)  # (C → C)

        # Branch 1: SSM path — regular (non-causal) 3D conv + Mamba scan
        self.conv_ssm = nn.Conv3d(half, half, kernel_size=3, padding=1, groups=half)
        self.mamba = _TriDirMambaOrFallback(half, d_state)

        # Branch 2: symmetric conv-only path (no SSM)
        self.conv_sym = nn.Conv3d(half, half, kernel_size=3, padding=1, groups=half)

        # Output projection merges both branches back to C channels
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)

        self.norm = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        residual = x
        x = self.norm(x)
        xp = self.in_proj(x)               # (B, C, D, H, W)

        # Split into two half-channel branches
        x1, x2 = xp.chunk(2, dim=1)       # each (B, C/2, D, H, W)

        # Branch 1 — SSM path
        x1 = F.silu(self.conv_ssm(x1))
        x1 = self.mamba(x1)                # TriDirectionalMamba or identity

        # Branch 2 — symmetric conv path (no SSM)
        x2 = F.silu(self.conv_sym(x2))

        # Concat and project
        out = self.out_proj(torch.cat([x1, x2], dim=1))
        return out + residual              # residual connection


class _TriDirMambaOrFallback(nn.Module):
    """
    Wraps TriDirectionalMamba with a graceful fallback to identity if the
    mamba_ssm / zeta package is not installed in the environment.
    """
    def __init__(self, channels: int, d_state: int):
        super().__init__()
        try:
            from utils.mamba_3d import TriDirectionalMamba
            self.mamba = TriDirectionalMamba(channels, d_state)
            self._use_mamba = True
        except Exception:
            # mamba_ssm not installed — fall back to a depthwise conv
            self.mamba = nn.Conv3d(channels, channels, kernel_size=3,
                                   padding=1, groups=channels)
            self._use_mamba = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mamba(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Windowed 3D Self-Attention  (used in final decoder stages)
# ─────────────────────────────────────────────────────────────────────────────

class WindowedSelfAttention3D(nn.Module):
    """
    Lightweight windowed multi-head self-attention for 3D volumes.

    The volume is partitioned into non-overlapping windows of size
    (ws, ws, ws). Attention is computed within each window independently,
    keeping memory tractable for volumetric data.

    Args:
        channels:   Number of feature channels.
        num_heads:  Number of attention heads.
        window_size: Spatial size of each attention window (default 4).
    """

    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 4):
        super().__init__()
        self.ws = window_size
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W)
        B, C, D, H, W = x.shape
        ws = self.ws
        residual = x

        # Pad so that D, H, W are divisible by ws
        pad_d = (ws - D % ws) % ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        _, _, Dp, Hp, Wp = x.shape

        # Partition into windows: (B, C, nD, ws, nH, ws, nW, ws)
        nD, nH, nW = Dp // ws, Hp // ws, Wp // ws
        x = x.view(B, C, nD, ws, nH, ws, nW, ws)
        # (B, nD, nH, nW, ws, ws, ws, C)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        # (B*nD*nH*nW, ws^3, C)
        num_windows = nD * nH * nW
        x = x.view(B * num_windows, ws * ws * ws, C)

        # Self-attention within each window
        xn = self.norm(x)
        attn_out, _ = self.attn(xn, xn, xn)
        x = x + attn_out   # residual inside window

        # Reconstruct spatial volume
        x = x.view(B, nD, nH, nW, ws, ws, ws, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, C, nD * ws, nH * ws, nW * ws)

        # Remove padding
        x = x[:, :, :D, :H, :W]

        out = self.proj(x)
        return out + residual


# ─────────────────────────────────────────────────────────────────────────────
# 3. Encoder block  (MambaVision mixer + CSSE3D)
# ─────────────────────────────────────────────────────────────────────────────

class MambaVisionEncoderBlock(nn.Module):
    """
    Encoder building block:
        Multi-scale conv (3,5,7,9) → MambaVision3DMixer → CSSE3D attention
    Keeps the multi-scale conv front-end from Model_MTASP for compatibility.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 d_state: int = 16):
        super().__init__()
        # Multi-scale parallel convs (same as ResidualMambaBlock in models.py)
        self.conv3 = nn.Conv3d(in_ch, out_ch, 3, stride, 1)
        self.conv5 = nn.Conv3d(in_ch, out_ch, 5, stride, 2)
        self.conv7 = nn.Conv3d(in_ch, out_ch, 7, stride, 3)
        self.conv9 = nn.Conv3d(in_ch, out_ch, 9, stride, 4)

        self.shortcut = (nn.Conv3d(in_ch, out_ch, 1, stride)
                         if in_ch != out_ch or stride != 1 else None)

        # MambaVision mixer for long-range context
        self.mixer = MambaVision3DMixer(out_ch, d_state)

        # Channel-Spatial SE for local recalibration
        self.csse = ChannelSpatialSELayer3D(out_ch)

        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (self.act(self.conv3(x)) + self.act(self.conv5(x)) +
               self.act(self.conv7(x)) + self.act(self.conv9(x)))
        out = self.mixer(out)
        out = self.csse(out)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. Decoder block  (MambaVision mixer OR self-attention depending on stage)
# ─────────────────────────────────────────────────────────────────────────────

class MambaVisionDecoderBlock(nn.Module):
    """
    Decoder building block.

    For the two final (highest-resolution) stages `use_attention=True` inserts
    a WindowedSelfAttention3D layer after the multi-scale convs — mirroring
    MambaVision's finding that self-attention in the last N/2 layers best
    recovers global context.

    For earlier decoder stages the MambaVision3DMixer is used instead.
    """

    def __init__(self, channels: int, d_state: int = 16,
                 use_attention: bool = False, num_heads: int = 4,
                 window_size: int = 4):
        super().__init__()
        self.conv3 = nn.Conv3d(channels, channels, 3, 1, 1)
        self.conv5 = nn.Conv3d(channels, channels, 5, 1, 2)
        self.conv7 = nn.Conv3d(channels, channels, 7, 1, 3)
        self.conv9 = nn.Conv3d(channels, channels, 9, 1, 4)

        self.use_attention = use_attention
        if use_attention:
            self.token_mixer = WindowedSelfAttention3D(
                channels, num_heads=num_heads, window_size=window_size)
        else:
            self.token_mixer = MambaVision3DMixer(channels, d_state)

        self.csse = ChannelSpatialSELayer3D(channels)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (self.act(self.conv3(x)) + self.act(self.conv5(x)) +
               self.act(self.conv7(x)) + self.act(self.conv9(x)))
        out = self.token_mixer(out)
        out = self.csse(out)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. Shared sub-modules (PI and AF) — identical to Model_MTASP
# ─────────────────────────────────────────────────────────────────────────────

class Comb_MV(nn.Module):
    """PTV Integration module (PI) — same design as in models.py."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_3x3 = nn.Conv3d(in_channels, in_channels, 3, padding=1)
        self.conv_5x5 = nn.Conv3d(in_channels, in_channels, 5, padding=2)
        self.conv_7x7 = nn.Conv3d(in_channels, in_channels, 7, padding=3)
        self.att_3x3  = nn.Conv3d(in_channels, in_channels, 1)
        self.att_5x5  = nn.Conv3d(in_channels, in_channels, 1)
        self.att_7x7  = nn.Conv3d(in_channels, in_channels, 1)
        self.att_fusion = nn.Conv3d(in_channels, 1, 1)
        self.channel_expand = nn.Conv3d(1, in_channels, 1)
        self.final_out = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, enc_out: torch.Tensor,
                ptv: torch.Tensor) -> torch.Tensor:
        ptv_r = F.interpolate(ptv, size=enc_out.shape[2:],
                              mode='trilinear', align_corners=False)
        if ptv_r.shape[1] != enc_out.shape[1]:
            ptv_r = self.channel_expand(ptv_r)

        x3 = F.relu(self.conv_3x3(enc_out));  p3 = F.relu(self.conv_3x3(ptv_r))
        x5 = F.relu(self.conv_5x5(enc_out));  p5 = F.relu(self.conv_5x5(ptv_r))
        x7 = F.relu(self.conv_7x7(enc_out));  p7 = F.relu(self.conv_7x7(ptv_r))

        a3 = torch.sigmoid(self.att_3x3(x3))
        a5 = torch.sigmoid(self.att_5x5(x5))
        a7 = torch.sigmoid(self.att_7x7(x7))

        fused = ((x3*a3 + p3*a3) + (x5*a5 + p5*a5) + (x7*a7 + p7*a7))
        fused = fused * torch.sigmoid(self.att_fusion(fused))
        return self.final_out(fused)


class AddFeatureMaps_MV(nn.Module):
    """Attention Fusion module (AF) — same design as in models.py."""

    def __init__(self, channels: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.ones(1))
        self.w2 = nn.Parameter(torch.ones(1))
        self.attention = ChannelSpatialSELayer3D(channels)

    def forward(self, F1: torch.Tensor, F2: torch.Tensor) -> torch.Tensor:
        return self.w1 * self.attention(F1) + self.w2 * self.attention(F2)


class UpConv_MV(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2,
                                       mode='trilinear', align_corners=True))


# ─────────────────────────────────────────────────────────────────────────────
# 6. Encoder
# ─────────────────────────────────────────────────────────────────────────────

class Encoder_MV(nn.Module):
    def __init__(self, in_ch: int, list_ch: list, d_state: int = 16):
        super().__init__()
        # 5 encoder stages  (ch: list_ch[1..5])
        self.enc1 = MambaVisionEncoderBlock(in_ch,       list_ch[1], stride=1, d_state=d_state)
        self.enc2 = MambaVisionEncoderBlock(list_ch[1],  list_ch[2], stride=2, d_state=d_state)
        self.enc3 = MambaVisionEncoderBlock(list_ch[2],  list_ch[3], stride=2, d_state=d_state)
        self.enc4 = MambaVisionEncoderBlock(list_ch[3],  list_ch[4], stride=2, d_state=d_state)
        self.enc5 = MambaVisionEncoderBlock(list_ch[4],  list_ch[5], stride=2, d_state=d_state)

        # PTV Integration (PI) from stage 2 onward
        self.comb2 = Comb_MV(list_ch[1], list_ch[1])
        self.comb3 = Comb_MV(list_ch[2], list_ch[2])
        self.comb4 = Comb_MV(list_ch[3], list_ch[3])
        self.comb5 = Comb_MV(list_ch[4], list_ch[4])

    def forward(self, x: torch.Tensor):
        ptv = x[:, 0:1]   # PTV channel

        e1 = self.enc1(x)
        e2 = self.enc2(self.comb2(e1, ptv))
        e3 = self.enc3(self.comb3(e2, ptv))
        e4 = self.enc4(self.comb4(e3, ptv))
        e5 = self.enc5(self.comb5(e4, ptv))

        return [e1, e2, e3, e4, e5]


# ─────────────────────────────────────────────────────────────────────────────
# 7. Decoder  (self-attention in stages 1 & 2 — the final/highest-res stages)
# ─────────────────────────────────────────────────────────────────────────────

class Decoder_MV(nn.Module):
    def __init__(self, list_ch: list, d_state: int = 16):
        super().__init__()
        # Up-convolutions
        self.up4 = UpConv_MV(list_ch[5], list_ch[4])
        self.up3 = UpConv_MV(list_ch[4], list_ch[3])
        self.up2 = UpConv_MV(list_ch[3], list_ch[2])
        self.up1 = UpConv_MV(list_ch[2], list_ch[1])

        # Attention Fusion (AF) skip connections
        self.af4 = AddFeatureMaps_MV(list_ch[4])
        self.af3 = AddFeatureMaps_MV(list_ch[3])
        self.af2 = AddFeatureMaps_MV(list_ch[2])
        self.af1 = AddFeatureMaps_MV(list_ch[1])

        # Decoder blocks:
        #   stages 4 & 3  → MambaVision mixer  (deeper, smaller spatial size)
        #   stages 2 & 1  → self-attention      (final, full-resolution stages)
        self.dec4 = MambaVisionDecoderBlock(list_ch[4], d_state=d_state,
                                            use_attention=False)
        self.dec3 = MambaVisionDecoderBlock(list_ch[3], d_state=d_state,
                                            use_attention=False)
        self.dec2 = MambaVisionDecoderBlock(list_ch[2], d_state=d_state,
                                            use_attention=True,
                                            num_heads=4, window_size=4)
        self.dec1 = MambaVisionDecoderBlock(list_ch[1], d_state=d_state,
                                            use_attention=True,
                                            num_heads=4, window_size=8)

    def forward(self, enc_feats):
        e1, e2, e3, e4, e5 = enc_feats

        d4 = self.dec4(self.af4(self.up4(e5), e4))
        d3 = self.dec3(self.af3(self.up3(d4),  e3))
        d2 = self.dec2(self.af2(self.up2(d3),  e2))
        d1 = self.dec1(self.af1(self.up1(d2),  e1))

        return d1


# ─────────────────────────────────────────────────────────────────────────────
# 8. Top-level model
# ─────────────────────────────────────────────────────────────────────────────

class Model_RANDose_MambaVision(nn.Module):
    """
    RANDose with MambaVision hybrid backbone.

    Encoder: MambaVision3DMixer blocks (SSM + symmetric conv branch)
             with PI (PTV Integration) from stage 2 onward.
    Decoder: MambaVision mixer in deep stages; windowed self-attention
             in the final two (highest-resolution) stages, following
             MambaVision's hybrid pattern (last N/2 layers = self-attention).
             AF (Attention Fusion) skip connections throughout.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 list_ch_A: list,
                 list_ch_B: list = None,   # kept for API compatibility
                 d_state: int = 16,
                 d_conv: int = 4,          # kept for API compatibility
                 expand: int = 2,          # kept for API compatibility
                 channel_token: bool = False):
        super().__init__()
        self.encoder = Encoder_MV(in_ch, list_ch_A, d_state)
        self.decoder = Decoder_MV(list_ch_A, d_state)
        self.out_conv = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return self.out_conv(dec)


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = Model_RANDose_MambaVision(
        in_ch=9, out_ch=1,
        list_ch_A=[-1, 16, 32, 64, 128, 256],
        d_state=16,
    )
    x = torch.randn(1, 9, 128, 128, 128)
    with torch.no_grad():
        y = model(x)
    print('Output shape:', y.shape)
    params = sum(p.numel() for p in model.parameters()) * 4e-6
    print(f'Param size: {params:.2f} MB')
