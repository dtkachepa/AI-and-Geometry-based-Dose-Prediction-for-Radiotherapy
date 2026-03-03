import torch
import torch.nn as nn
from torch import Tensor


class TriDirectionalMamba(nn.Module):
    """
    Tri-directional Mamba scanning for 3D medical volumes.

    Processes the input feature map (B, C, D, H, W) along three anatomical
    scanning directions and fuses the results with learned weights:

        F_Mamba = softmax(α) · [F_axial, F_coronal, F_sagittal]   (eq. 10)

    Scanning directions:
        Axial   (z-axis): for each (h, w) position, scan the depth sequence D
        Coronal (y-axis): for each (d, w) position, scan the height sequence H
        Sagittal(x-axis): for each (d, h) position, scan the width  sequence W

    This gives the Mamba SSM access to long-range dependencies in all three
    anatomical orientations while remaining linear in sequence length.

    Args:
        channels: Number of feature channels (C). In/out channels are the same.
        d_state:  Mamba state dimension (controls SSM expressiveness).
    """

    def __init__(self, channels: int, d_state: int = 16):
        super().__init__()
        from zeta.nn import MambaBlock

        # One Mamba block per scanning direction (depth=1 layer each)
        self.mamba_z = MambaBlock(channels, 1, d_state)
        self.mamba_y = MambaBlock(channels, 1, d_state)
        self.mamba_x = MambaBlock(channels, 1, d_state)

        # Pre-norm for each direction (applied before the Mamba block)
        self.norm_z = nn.LayerNorm(channels)
        self.norm_y = nn.LayerNorm(channels)
        self.norm_x = nn.LayerNorm(channels)

        # Learnable fusion weights α1, α2, α3 (initialised equally)
        self.alpha = nn.Parameter(torch.ones(3))

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, D, H, W) feature map

        Returns:
            (B, C, D, H, W) — same shape, long-range context captured
        """
        B, C, D, H, W = x.shape

        # ── Axial scan (z-axis) ─────────────────────────────────────────────
        # For each (h, w) pixel, form a length-D sequence along depth.
        # (B, C, D, H, W) → (B, H, W, D, C) → (B*H*W, D, C)
        x_z = x.permute(0, 3, 4, 2, 1).contiguous().reshape(B * H * W, D, C)
        x_z = self.mamba_z(self.norm_z(x_z))
        # (B*H*W, D, C) → (B, H, W, D, C) → (B, C, D, H, W)
        x_z = x_z.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2).contiguous()

        # ── Coronal scan (y-axis) ───────────────────────────────────────────
        # For each (d, w) pixel, form a length-H sequence along height.
        # (B, C, D, H, W) → (B, D, W, H, C) → (B*D*W, H, C)
        x_y = x.permute(0, 2, 4, 3, 1).contiguous().reshape(B * D * W, H, C)
        x_y = self.mamba_y(self.norm_y(x_y))
        # (B*D*W, H, C) → (B, D, W, H, C) → (B, C, D, H, W)
        x_y = x_y.reshape(B, D, W, H, C).permute(0, 4, 1, 3, 2).contiguous()

        # ── Sagittal scan (x-axis) ──────────────────────────────────────────
        # For each (d, h) pixel, form a length-W sequence along width.
        # (B, C, D, H, W) → (B, D, H, W, C) → (B*D*H, W, C)
        x_x = x.permute(0, 2, 3, 4, 1).contiguous().reshape(B * D * H, W, C)
        x_x = self.mamba_x(self.norm_x(x_x))
        # (B*D*H, W, C) → (B, D, H, W, C) → (B, C, D, H, W)
        x_x = x_x.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()

        # ── Weighted fusion (eq. 10) ────────────────────────────────────────
        weights = torch.softmax(self.alpha, dim=0)
        return weights[0] * x_z + weights[1] * x_y + weights[2] * x_x
