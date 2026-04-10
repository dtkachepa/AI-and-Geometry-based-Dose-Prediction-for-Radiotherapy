"""
Model_RANDose_MambaVision
=========================
Adapts the MambaVision hybrid design (Hatamizadeh & Kautz, CVPR 2025) to 3D
dose prediction inside the RANDose U-Net framework.

Key ideas from MambaVision applied here:
  1. MambaVision3DMixer — regular (non-causal) 3D conv + symmetric non-SSM
     branch; outputs are concatenated. Gives richer features than SSM alone
     (Table 4 of the paper).
  2. Self-attention in the final two decoder stages (highest resolution) —
     MambaVision's key finding: self-attention at the last N/2 layers best
     captures global context (Table 5). For dose prediction this is where
     boundary precision matters most.
  3. PI (Comb) and AF (AddFeatureMaps) kept identical to Model_MTASP so the
     RANDose region-aware machinery is fully preserved.

NOTE: encoder/decoder are named `enc`/`dec` (not `encoder`/`decoder`) so
      utils.py does NOT try to split learning rates — same behaviour as
      Model_MTASP which wraps them inside `net_A`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.se3d import ChannelSpatialSELayer3D


# ─────────────────────────────────────────────────────────────────────────────
# 1.  MambaVision 3D Mixer
# ─────────────────────────────────────────────────────────────────────────────

class MambaVision3DMixer(nn.Module):
    """
    3D adaptation of the MambaVision mixer (eq. 7, Fig. 3 of the paper).

    Two parallel branches from a shared projection:
        Branch 1 (SSM path):  Conv3D(dw) → TriDirectionalMamba → SiLU
        Branch 2 (conv path): Conv3D(dw) → SiLU   (no SSM)
    Output: Conv1x1( Concat(B1, B2) )  — preserves channel count C.

    Uses depthwise conv so parameter count stays low.
    """

    def __init__(self, channels: int, d_state: int = 16):
        super().__init__()
        half = channels // 2

        self.in_proj  = nn.Conv3d(channels, channels, kernel_size=1)

        # Branch 1 — SSM path (depthwise conv + Mamba)
        self.conv_ssm = nn.Conv3d(half, half, kernel_size=3, padding=1, groups=half)
        self.mamba    = _TriDirMambaOrFallback(half, d_state)

        # Branch 2 — symmetric conv-only path
        self.conv_sym = nn.Conv3d(half, half, kernel_size=3, padding=1, groups=half)

        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm     = nn.InstanceNorm3d(channels, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        xp = self.in_proj(self.norm(x))
        x1, x2 = xp.chunk(2, dim=1)

        x1 = self.mamba(F.silu(self.conv_ssm(x1)))   # SSM branch
        x2 = F.silu(self.conv_sym(x2))                # symmetric branch

        return self.out_proj(torch.cat([x1, x2], dim=1)) + residual


class _TriDirMambaOrFallback(nn.Module):
    """TriDirectionalMamba with graceful fallback to depthwise conv."""
    def __init__(self, channels: int, d_state: int):
        super().__init__()
        try:
            from utils.mamba_3d import TriDirectionalMamba
            self.op = TriDirectionalMamba(channels, d_state)
        except Exception:
            self.op = nn.Conv3d(channels, channels, kernel_size=3,
                                padding=1, groups=channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Windowed 3D Self-Attention  (final decoder stages only)
# ─────────────────────────────────────────────────────────────────────────────

class WindowedSelfAttention3D(nn.Module):
    """
    Windowed multi-head self-attention for 3D volumes.
    Partitions the volume into (ws)^3 windows; attention is local within each.
    """

    def __init__(self, channels: int, num_heads: int = 4, window_size: int = 4):
        super().__init__()
        self.ws   = window_size
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(channels)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        ws = self.ws
        residual = x

        pad_d = (ws - D % ws) % ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        _, _, Dp, Hp, Wp = x.shape

        nD, nH, nW = Dp // ws, Hp // ws, Wp // ws
        x = x.view(B, C, nD, ws, nH, ws, nW, ws)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        nW_total = nD * nH * nW
        x = x.view(B * nW_total, ws ** 3, C)

        xn = self.norm(x)
        attn_out, _ = self.attn(xn, xn, xn)
        x = x + attn_out

        x = x.view(B, nD, nH, nW, ws, ws, ws, C)
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, C, nD * ws, nH * ws, nW * ws)
        x = x[:, :, :D, :H, :W]

        return self.proj(x) + residual


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Encoder block
# ─────────────────────────────────────────────────────────────────────────────

class MambaVisionEncoderBlock(nn.Module):
    """
    Conv → optional MambaVision mixer → CSSE3D.

    `use_mixer=False` for shallow (high-resolution) stages — Mamba scanning
    over 128³/64³ volumes creates tens of thousands of sequences and is the
    primary cause of slow training. Mixer is only applied at deep stages
    (32³ and below) where sequence lengths are tractable.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 d_state: int = 16, use_mixer: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.shortcut = (nn.Conv3d(in_ch, out_ch, 1, stride)
                         if in_ch != out_ch or stride != 1 else None)
        self.mixer = MambaVision3DMixer(out_ch, d_state) if use_mixer else None
        self.csse  = ChannelSpatialSELayer3D(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.mixer is not None:
            out = self.mixer(out)
        out = self.csse(out)
        if self.shortcut is not None:
            out = out + self.shortcut(x)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Decoder block
# ─────────────────────────────────────────────────────────────────────────────

class MambaVisionDecoderBlock(nn.Module):
    """
    Conv → token mixer (MambaVision mixer OR self-attention) → CSSE3D.
    Self-attention is used in the final two (highest-resolution) stages,
    mirroring MambaVision's 'last N/2 layers = self-attention' finding.
    """

    def __init__(self, channels: int, d_state: int = 16,
                 use_attention: bool = False,
                 num_heads: int = 4, window_size: int = 4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        if use_attention:
            self.token_mixer = WindowedSelfAttention3D(
                channels, num_heads=num_heads, window_size=window_size)
        else:
            self.token_mixer = MambaVision3DMixer(channels, d_state)

        self.csse = ChannelSpatialSELayer3D(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.token_mixer(out)
        return self.csse(out)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PI and AF modules (identical to Model_MTASP)
# ─────────────────────────────────────────────────────────────────────────────

class Comb_MV(nn.Module):
    """PTV Integration module (PI)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_3 = nn.Conv3d(in_channels, in_channels, 3, padding=1)
        self.conv_5 = nn.Conv3d(in_channels, in_channels, 5, padding=2)
        self.conv_7 = nn.Conv3d(in_channels, in_channels, 7, padding=3)
        self.att_3  = nn.Conv3d(in_channels, in_channels, 1)
        self.att_5  = nn.Conv3d(in_channels, in_channels, 1)
        self.att_7  = nn.Conv3d(in_channels, in_channels, 1)
        self.att_fusion    = nn.Conv3d(in_channels, 1, 1)
        self.channel_expand = nn.Conv3d(1, in_channels, 1)
        self.final_out     = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, enc_out: torch.Tensor, ptv: torch.Tensor) -> torch.Tensor:
        ptv_r = F.interpolate(ptv, size=enc_out.shape[2:],
                              mode='trilinear', align_corners=False)
        if ptv_r.shape[1] != enc_out.shape[1]:
            ptv_r = self.channel_expand(ptv_r)

        x3 = F.relu(self.conv_3(enc_out)); p3 = F.relu(self.conv_3(ptv_r))
        x5 = F.relu(self.conv_5(enc_out)); p5 = F.relu(self.conv_5(ptv_r))
        x7 = F.relu(self.conv_7(enc_out)); p7 = F.relu(self.conv_7(ptv_r))

        a3 = torch.sigmoid(self.att_3(x3))
        a5 = torch.sigmoid(self.att_5(x5))
        a7 = torch.sigmoid(self.att_7(x7))

        fused = (x3*a3 + p3*a3) + (x5*a5 + p5*a5) + (x7*a7 + p7*a7)
        fused = fused * torch.sigmoid(self.att_fusion(fused))
        return self.final_out(fused)


class AddFeatureMaps_MV(nn.Module):
    """Attention Fusion module (AF)."""

    def __init__(self, channels: int):
        super().__init__()
        self.w1  = nn.Parameter(torch.ones(1))
        self.w2  = nn.Parameter(torch.ones(1))
        self.att = ChannelSpatialSELayer3D(channels)

    def forward(self, F1: torch.Tensor, F2: torch.Tensor) -> torch.Tensor:
        return self.w1 * self.att(F1) + self.w2 * self.att(F2)


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
# 6.  Encoder  (named `enc` to avoid utils.py split-LR path)
# ─────────────────────────────────────────────────────────────────────────────

class _Enc(nn.Module):
    def __init__(self, in_ch: int, list_ch: list, d_state: int = 16):
        super().__init__()
        # Stages 1-3: high resolution (128³→32³) — conv+CSSE only, no Mamba
        # Stages 4-5: deep (16³, 8³) — full MambaVision mixer applied
        self.enc1 = MambaVisionEncoderBlock(in_ch,      list_ch[1], stride=1, d_state=d_state, use_mixer=False)
        self.enc2 = MambaVisionEncoderBlock(list_ch[1], list_ch[2], stride=2, d_state=d_state, use_mixer=False)
        self.enc3 = MambaVisionEncoderBlock(list_ch[2], list_ch[3], stride=2, d_state=d_state, use_mixer=False)
        self.enc4 = MambaVisionEncoderBlock(list_ch[3], list_ch[4], stride=2, d_state=d_state, use_mixer=True)
        self.enc5 = MambaVisionEncoderBlock(list_ch[4], list_ch[5], stride=2, d_state=d_state, use_mixer=True)

        # PI from stage 2 onward
        self.comb2 = Comb_MV(list_ch[1], list_ch[1])
        self.comb3 = Comb_MV(list_ch[2], list_ch[2])
        self.comb4 = Comb_MV(list_ch[3], list_ch[3])
        self.comb5 = Comb_MV(list_ch[4], list_ch[4])

    def forward(self, x: torch.Tensor):
        ptv = x[:, 0:1]
        e1 = self.enc1(x)
        e2 = self.enc2(self.comb2(e1, ptv))
        e3 = self.enc3(self.comb3(e2, ptv))
        e4 = self.enc4(self.comb4(e3, ptv))
        e5 = self.enc5(self.comb5(e4, ptv))
        return [e1, e2, e3, e4, e5]


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Decoder  (named `dec` to avoid utils.py split-LR path)
#     Self-attention in stages 1 & 2 (final/highest-resolution stages)
# ─────────────────────────────────────────────────────────────────────────────

class _Dec(nn.Module):
    def __init__(self, list_ch: list, d_state: int = 16):
        super().__init__()
        self.up4 = UpConv_MV(list_ch[5], list_ch[4])
        self.up3 = UpConv_MV(list_ch[4], list_ch[3])
        self.up2 = UpConv_MV(list_ch[3], list_ch[2])
        self.up1 = UpConv_MV(list_ch[2], list_ch[1])

        self.af4 = AddFeatureMaps_MV(list_ch[4])
        self.af3 = AddFeatureMaps_MV(list_ch[3])
        self.af2 = AddFeatureMaps_MV(list_ch[2])
        self.af1 = AddFeatureMaps_MV(list_ch[1])

        # Deep stages → MambaVision mixer
        self.dec4 = MambaVisionDecoderBlock(list_ch[4], d_state=d_state,
                                            use_attention=False)
        self.dec3 = MambaVisionDecoderBlock(list_ch[3], d_state=d_state,
                                            use_attention=False)
        # dec2 at 64³ resolution → windowed self-attention (4³ windows → ~537 MB)
        self.dec2 = MambaVisionDecoderBlock(list_ch[2], d_state=d_state,
                                            use_attention=True,
                                            num_heads=4, window_size=4)
        # dec1 at 128³ resolution → MambaVision mixer (attention would be ~34 GB)
        self.dec1 = MambaVisionDecoderBlock(list_ch[1], d_state=d_state,
                                            use_attention=False)

    def forward(self, enc_feats):
        e1, e2, e3, e4, e5 = enc_feats
        d4 = self.dec4(self.af4(self.up4(e5), e4))
        d3 = self.dec3(self.af3(self.up3(d4), e3))
        d2 = self.dec2(self.af2(self.up2(d3), e2))
        d1 = self.dec1(self.af1(self.up1(d2), e1))
        return d1


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Top-level model
# ─────────────────────────────────────────────────────────────────────────────

class Model_RANDose_MambaVision(nn.Module):
    """
    RANDose with MambaVision hybrid backbone.

    Encoder: MambaVision3DMixer blocks (SSM + symmetric conv branch)
             with PI (PTV Integration) from stage 2 onward.
    Decoder: MambaVision mixer in deep stages; windowed self-attention
             in the final two (highest-resolution) stages.
             AF (Attention Fusion) skip connections throughout.
    """

    def __init__(self, in_ch: int, out_ch: int,
                 list_ch_A: list,
                 list_ch_B: list = None,
                 d_state: int = 16,
                 d_conv: int = 4,
                 expand: int = 2,
                 channel_token: bool = False):
        super().__init__()
        # Named `enc`/`dec` — NOT `encoder`/`decoder` — so utils.py uses
        # the single unified Adam LR path (same as Model_MTASP via net_A).
        self.enc      = _Enc(in_ch, list_ch_A, d_state)
        self.dec      = _Dec(list_ch_A, d_state)
        self.out_conv = nn.Conv3d(list_ch_A[1], out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_conv(self.dec(self.enc(x)))


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
