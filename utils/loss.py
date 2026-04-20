import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt, PTVs):
        pred_A = pred
        gt_dose = gt[0]
        possible_dose_mask = gt[1]


        pred_A = pred_A[possible_dose_mask > 0]

        gt_dose = gt_dose[possible_dose_mask > 0]

        L1_loss = self.L1_loss_func(pred_A, gt_dose)
        return L1_loss
    
import torch
import torch.nn as nn
from pytorch_msssim import ssim


    
class Loss_DC(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

        # Define learnable weights for max and min dose penalties
        self.max_dose_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.min_dose_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)


    def forward(self, pred, gt, PTVs):

        device = pred.device
        pred_A = pred
        gt_dose = gt[0]
        possible_dose_mask = gt[1]

        
       
        # Mask the predicted and ground truth values
        pred_A = pred_A[possible_dose_mask > 0]
        gt_dose = gt_dose[possible_dose_mask > 0]

        # L1 loss
        L1_loss = self.L1_loss_func(pred_A, gt_dose)

        # Dynamic Dose Constraints (penalizing extreme doses)
        max_dose_limit = gt_dose.max()
        min_dose_limit = gt_dose.min()

        # Squared penalties for extreme doses
        max_dose_penalty = torch.clamp(pred_A.max() - max_dose_limit, min=0) ** 2
        min_dose_penalty = torch.clamp(min_dose_limit - pred_A.min(), min=0) ** 2

        # Apply learnable weights for dose constraint penalties
        dose_constraint_loss = self.max_dose_weight * max_dose_penalty + self.min_dose_weight * min_dose_penalty
        
        # Total loss
        total_loss = L1_loss + dose_constraint_loss 

        return total_loss
    

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvancedLoss(nn.Module):
    def __init__(self):
        super(AdvancedLoss, self).__init__()

        # Initialize learnable parameters for each loss weight
        self.alpha = nn.Parameter(torch.tensor(1.0))  # Weight for L1 loss
        self.beta = nn.Parameter(torch.tensor(1.0))   # Weight for smoothness loss

        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt, PTV, OAR, mask=None):
        # Extract ground truth components
        gt_dose = gt[0]  # GT dose values
        possible_dose_mask = gt[1]  # Mask for valid dose regions
    
        # Mask the predictions and ground truth based on valid regions
        pred_masked = pred[possible_dose_mask > 0]
        gt_dose_masked = gt_dose[possible_dose_mask > 0]

        # L1 Loss
        L1_loss = self.L1_loss_func(pred_masked, gt_dose_masked)

        # Edge-Aware Smoothness Regularization
        smoothness_loss = self.compute_smoothness_loss(pred, gt_dose, mask)



        # Weighted sum of all losses using learnable weights
        total_loss = (self.alpha * L1_loss +
                      self.beta * smoothness_loss)

        return total_loss

    def compute_smoothness_loss(self, pred, gt_dose, mask):
        # Compute gradients along the x, y, z directions
        gradient_x = torch.abs(pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :])
        gradient_y = torch.abs(pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :])
        gradient_z = torch.abs(pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1])

        # Edge mask based on ground truth dose gradients
        edge_mask_x = torch.abs(gt_dose[:, :, 1:, :, :] - gt_dose[:, :, :-1, :, :])
        edge_mask_y = torch.abs(gt_dose[:, :, :, 1:, :] - gt_dose[:, :, :, :-1, :])
        edge_mask_z = torch.abs(gt_dose[:, :, :, :, 1:] - gt_dose[:, :, :, :, :-1])

        # Penalize gradients where there are no edges
        smoothness_loss = (gradient_x * (1 - edge_mask_x)).mean() + \
                           (gradient_y * (1 - edge_mask_y)).mean() + \
                           (gradient_z * (1 - edge_mask_z)).mean()

        return smoothness_loss


class SharpDoseLoss(nn.Module):
    def __init__(self):
        super(SharpDoseLoss, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))    
        self.beta = nn.Parameter(torch.tensor(2.0))     
        self.gamma = nn.Parameter(torch.tensor(1.5))    
        
        self.L1_loss_func = nn.L1Loss(reduction='mean')
        
    def forward(self, pred, gt, PTV, OAR, mask=None):
        gt_dose = gt[0]
        possible_dose_mask = gt[1]
        
        # Basic L1 Loss
        pred_masked = pred[possible_dose_mask > 0]
        gt_masked = gt_dose[possible_dose_mask > 0]
        L1_loss = self.L1_loss_func(pred_masked, gt_masked)
        
        # Enhanced gradient matching with focus on sharp transitions
        gradient_loss = self.compute_sharp_gradient_loss(pred, gt_dose)
        
        # Special focus on high gradient regions
        high_gradient_loss = self.compute_high_gradient_region_loss(pred, gt_dose)
        
        total_loss = (self.alpha * L1_loss + 
                     self.beta * gradient_loss +
                     self.gamma * high_gradient_loss)
        
        return total_loss
    
    def compute_sharp_gradient_loss(self, pred, gt_dose):
        # Compute gradients in all directions
        dx_pred = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        dy_pred = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        dz_pred = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        
        dx_gt = gt_dose[:, :, 1:, :, :] - gt_dose[:, :, :-1, :, :]
        dy_gt = gt_dose[:, :, :, 1:, :] - gt_dose[:, :, :, :-1, :]
        dz_gt = gt_dose[:, :, :, :, 1:] - gt_dose[:, :, :, :, :-1]
        
        gradient_loss = (torch.pow(torch.abs(dx_pred - dx_gt), 2.0).mean() +
                        torch.pow(torch.abs(dy_pred - dy_gt), 2.0).mean() +
                        torch.pow(torch.abs(dz_pred - dz_gt), 2.0).mean())
        
        return gradient_loss
    
    def compute_high_gradient_region_loss(self, pred, gt_dose):
        # Compute padded gradients to maintain size
        dx_gt = torch.zeros_like(gt_dose)
        dy_gt = torch.zeros_like(gt_dose)
        dz_gt = torch.zeros_like(gt_dose)
        
        dx_gt[:, :, 1:, :, :] = gt_dose[:, :, 1:, :, :] - gt_dose[:, :, :-1, :, :]
        dy_gt[:, :, :, 1:, :] = gt_dose[:, :, :, 1:, :] - gt_dose[:, :, :, :-1, :]
        dz_gt[:, :, :, :, 1:] = gt_dose[:, :, :, :, 1:] - gt_dose[:, :, :, :, :-1]
        
        # Calculate gradient magnitude
        gradient_magnitude = torch.sqrt(dx_gt**2 + dy_gt**2 + dz_gt**2)
        
        # Create mask for high gradient regions
        threshold = gradient_magnitude.mean() + gradient_magnitude.std()
        high_gradient_mask = (gradient_magnitude > threshold)
        
        # Calculate predicted gradients with padding
        dx_pred = torch.zeros_like(pred)
        dy_pred = torch.zeros_like(pred)
        dz_pred = torch.zeros_like(pred)
        
        dx_pred[:, :, 1:, :, :] = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        dy_pred[:, :, :, 1:, :] = pred[:, :, :, 1:, :] - pred[:, :, :, :-1, :]
        dz_pred[:, :, :, :, 1:] = pred[:, :, :, :, 1:] - pred[:, :, :, :, :-1]
        
        # Focus loss on high gradient regions
        high_gradient_loss = F.mse_loss(
            dx_pred[high_gradient_mask],
            dx_gt[high_gradient_mask]
        ) + F.mse_loss(
            dy_pred[high_gradient_mask],
            dy_gt[high_gradient_mask]
        ) + F.mse_loss(
            dz_pred[high_gradient_mask],
            dz_gt[high_gradient_mask]
        )
        
        return high_gradient_loss
    

class Loss_DC_PTV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

        # Define learnable weights for max and min dose penalties
        self.max_dose_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.min_dose_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # Weight for penalizing incorrect dose in PTV region
        self.PTV_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # Weight for penalizing incorrect dose in OAR region
        self.OAR_weights = nn.Parameter(torch.ones(7), requires_grad=True)  # One weight for each OAR

    def forward(self, pred, gt, PTVs, OAR):
        device = pred.device
        pred_A = pred
        gt_dose = gt[0]
        possible_dose_mask = gt[1]
        
        
        # Masked values using possible_dose_mask
        pred_A_masked = pred_A[possible_dose_mask > 0]
        gt_dose_masked = gt_dose[possible_dose_mask > 0]

        # ---- Standard L1 loss ----
        L1_loss = self.L1_loss_func(pred_A_masked, gt_dose_masked)

        # ---- PTV Weighted Loss ----
        # Extract values inside the PTV region
        pred_PTV = pred_A[PTVs > 0]  # Predicted doses within PTV
        gt_PTV = gt_dose[PTVs > 0]  # Ground truth doses within PTV

        PTV_Loss = self.L1_loss_func(pred_PTV, gt_PTV)  # L1 loss in PTV
        PTV_Loss = self.PTV_weight * PTV_Loss  # Scale by learnable weight

        # ---- OAR Combined Binary Mask ----
        # Create a combined binary mask for all OARs (any voxel in any OAR is set to 1)
        combined_OAR_mask = torch.sum(OAR, dim=1) > 0  # This creates a binary mask for all OARs combined
        
        # Make sure the combined_OAR_mask has the same shape as pred_A by adding an extra channel dimension (1)
        combined_OAR_mask = combined_OAR_mask.unsqueeze(1)  # Shape [batch_size, 1, height, width, depth]
        
        # Predicted doses inside the combined OAR region (mask applied to OAR region)
        pred_OAR_combined = pred_A[combined_OAR_mask > 0]  
        gt_OAR_combined = gt_dose[combined_OAR_mask > 0]  
        
        # L1 loss for the combined OAR region
        OAR_Loss = self.L1_loss_func(pred_OAR_combined, gt_OAR_combined)

        # ---- Dose Constraints (Penalizing Extreme Doses) ----
        max_dose_limit = gt_dose_masked.max()
        min_dose_limit = gt_dose_masked.min()

        max_dose_penalty = torch.clamp(pred_A_masked.max() - max_dose_limit, min=0) ** 2
        min_dose_penalty = torch.clamp(min_dose_limit - pred_A_masked.min(), min=0) ** 2

        dose_constraint_loss = self.max_dose_weight * max_dose_penalty + self.min_dose_weight * min_dose_penalty

        # ---- Total Loss ----
        total_loss = L1_loss + PTV_Loss + OAR_Loss + dose_constraint_loss

        return total_loss



class Loss_DC_PTV(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

        # Define learnable weights for max and min dose penalties
        self.max_dose_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.min_dose_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # Weight for penalizing incorrect dose in PTV region
        self.PTV_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)

        # Weight for penalizing incorrect dose in OAR region
        self.OAR_weights = nn.Parameter(torch.ones(7), requires_grad=True)  # One weight for each OAR

    def forward(self, pred, gt, PTVs, OAR):
        device = pred.device
        pred_A = pred
        gt_dose = gt[0]
        possible_dose_mask = gt[1]
        
        
        # Masked values using possible_dose_mask
        pred_A_masked = pred_A[possible_dose_mask > 0]
        gt_dose_masked = gt_dose[possible_dose_mask > 0]

        # ---- Standard L1 loss ----
        L1_loss = self.L1_loss_func(pred_A_masked, gt_dose_masked)

        # ---- PTV Weighted Loss ----
        # Extract values inside the PTV region
        pred_PTV = pred_A[PTVs > 0]  # Predicted doses within PTV
        gt_PTV = gt_dose[PTVs > 0]  # Ground truth doses within PTV

        PTV_Loss = self.L1_loss_func(pred_PTV, gt_PTV)  # L1 loss in PTV
        PTV_Loss = self.PTV_weight * PTV_Loss  # Scale by learnable weight

        # ---- OAR Combined Binary Mask ----
        # Create a combined binary mask for all OARs (any voxel in any OAR is set to 1)
        combined_OAR_mask = torch.sum(OAR, dim=1) > 0  # This creates a binary mask for all OARs combined
        
        # Make sure the combined_OAR_mask has the same shape as pred_A by adding an extra channel dimension (1)
        combined_OAR_mask = combined_OAR_mask.unsqueeze(1)  # Shape [batch_size, 1, height, width, depth]
        
        # Predicted doses inside the combined OAR region (mask applied to OAR region)
        pred_OAR_combined = pred_A[combined_OAR_mask > 0]  
        gt_OAR_combined = gt_dose[combined_OAR_mask > 0]  
        
        # L1 loss for the combined OAR region
        OAR_Loss = self.L1_loss_func(pred_OAR_combined, gt_OAR_combined)

        # ---- Total Loss ----
        total_loss = L1_loss + PTV_Loss + OAR_Loss
        
        # ---- Check if PTV and OAR regions contain non-zero values (i.e., if they exist) ----
        return total_loss


# ── Improvement 1: Boundary-Aware Loss ────────────────────────────────────────
#
# Extends Loss_DC_PTV with two new terms (proposal eqs. 5-7):
#
#   L_Boundary = w_boundary · (1/|B|) Σ_{i∈B} |ŷ_i − y_i|   (eq. 5)
#   L_Gradient = w_gradient · ‖∇ŷ − ∇y‖₁                    (eq. 6)
#   L_Total    = L_L1 + L_PTV + L_OAR + L_Boundary + L_Gradient  (eq. 7)
#
# Boundary set B is extracted from the PTV mask using either:
#   Method 1 — Morphological: B = M_PTV − (M_PTV ⊖ S_k)      (eq. 3)
#   Method 2 — Gradient-based: B = ‖∇M_PTV‖ > 0              (eq. 4)

class Loss_BoundaryAware(nn.Module):
    """
    Boundary-Aware Loss that adds explicit supervision at PTV surface voxels
    and enforces gradient consistency across the whole dose volume.

    Args:
        boundary_method : 'morph'  — morphological shell (eq. 3)
                          'grad'   — Sobel-gradient boundary (eq. 4)
        boundary_thickness : shell radius k for morphological method (default 2)
        w_boundary : initial weight for L_Boundary (learnable, default 1.0)
        w_gradient : initial weight for L_Gradient (learnable, default 0.5)
    """

    def __init__(self,
                 boundary_method: str = 'morph',
                 boundary_thickness: int = 2,
                 w_boundary: float = 1.0,
                 w_gradient: float = 0.5):
        super().__init__()

        self.boundary_method = boundary_method
        self.k = boundary_thickness

        # Base region weights (learnable, matching Loss_DC_PTV)
        self.PTV_weight = nn.Parameter(torch.tensor(1.0))
        self.OAR_weight = nn.Parameter(torch.tensor(1.0))

        # Boundary-specific weights (learnable — proposal §A.4)
        self.w_boundary = nn.Parameter(torch.tensor(w_boundary))
        self.w_gradient = nn.Parameter(torch.tensor(w_gradient))

        self.L1 = nn.L1Loss(reduction='mean')

        # ── Pre-build morphological structuring element (sphere of radius k) ──
        size = 2 * self.k + 1
        coords = torch.arange(size, dtype=torch.float32) - self.k
        zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing='ij')
        sphere = ((xx ** 2 + yy ** 2 + zz ** 2) <= self.k ** 2).float()
        # shape (1, 1, size, size, size) for F.conv3d
        self.register_buffer('sphere_kernel', sphere.unsqueeze(0).unsqueeze(0))
        self.sphere_sum = float(sphere.sum().item())

        # ── 3D Sobel kernels (registered as buffers → auto device transfer) ──
        # Gx: gradient along x (left-right)
        Gx = torch.tensor(
            [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
             [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
             [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]], dtype=torch.float32)
        # Gy: gradient along y (up-down)
        Gy = torch.tensor(
            [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
             [[-2, -4, -2], [0, 0, 0], [2, 4, 2]],
             [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]], dtype=torch.float32)
        # Gz: gradient along z (depth)
        Gz = torch.tensor(
            [[[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]],
             [[ 0,  0,  0], [ 0,  0,  0], [ 0,  0,  0]],
             [[ 1,  2,  1], [ 2,  4,  2], [ 1,  2,  1]]], dtype=torch.float32)
        self.register_buffer('Gx', Gx.unsqueeze(0).unsqueeze(0))
        self.register_buffer('Gy', Gy.unsqueeze(0).unsqueeze(0))
        self.register_buffer('Gz', Gz.unsqueeze(0).unsqueeze(0))

    # ── Boundary extraction ────────────────────────────────────────────────

    def _boundary_morph(self, ptv: torch.Tensor) -> torch.Tensor:
        """
        Morphological shell: B = M_PTV − (M_PTV ⊖ S_k)   (eq. 3)
        Returns a boolean mask of the same shape as ptv.
        """
        # Erosion via convolution: a voxel survives iff all sphere positions = 1
        eroded = F.conv3d(ptv.float(), self.sphere_kernel.to(ptv.device), padding=self.k)
        eroded = (eroded >= self.sphere_sum - 0.5).float()
        return (ptv.float() - eroded) > 0.5

    def _boundary_grad(self, ptv: torch.Tensor) -> torch.Tensor:
        """
        Gradient-based boundary: B = {‖∇M_PTV‖ > 0}   (eq. 4)
        Returns a boolean mask of the same shape as ptv.
        """
        m = ptv.float()
        Gx = self.Gx.to(m.device)
        Gy = self.Gy.to(m.device)
        Gz = self.Gz.to(m.device)
        gx = F.conv3d(m, Gx, padding=1)
        gy = F.conv3d(m, Gy, padding=1)
        gz = F.conv3d(m, Gz, padding=1)
        return (gx ** 2 + gy ** 2 + gz ** 2) > 0

    def _sobel_gradients(self, vol: torch.Tensor):
        """Return (gx, gy, gz) Sobel gradients of a dose volume."""
        Gx = self.Gx.to(vol.device)
        Gy = self.Gy.to(vol.device)
        Gz = self.Gz.to(vol.device)
        gx = F.conv3d(vol, Gx, padding=1)
        gy = F.conv3d(vol, Gy, padding=1)
        gz = F.conv3d(vol, Gz, padding=1)
        return gx, gy, gz

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, pred, gt, PTVs, OAR):
        gt_dose = gt[0]
        possible_dose_mask = gt[1]

        # ── L1 (global, within possible-dose region) ──────────────────────
        pred_m = pred[possible_dose_mask > 0]
        gt_m   = gt_dose[possible_dose_mask > 0]
        L1_loss = self.L1(pred_m, gt_m)

        # ── L_PTV ─────────────────────────────────────────────────────────
        ptv_vox = PTVs > 0
        L_ptv = (self.PTV_weight * self.L1(pred[ptv_vox], gt_dose[ptv_vox])
                 if ptv_vox.any() else pred.new_tensor(0.0))

        # ── L_OAR ─────────────────────────────────────────────────────────
        oar_mask = (OAR.sum(dim=1, keepdim=True) > 0)
        L_oar = (self.OAR_weight * self.L1(pred[oar_mask], gt_dose[oar_mask])
                 if oar_mask.any() else pred.new_tensor(0.0))

        # ── Boundary mask ─────────────────────────────────────────────────
        if self.boundary_method == 'grad':
            boundary_mask = self._boundary_grad(PTVs)
        else:
            boundary_mask = self._boundary_morph(PTVs)

        # ── L_Boundary (eq. 5) ────────────────────────────────────────────
        # Normalised by |B| (mean already does this)
        L_boundary = (self.w_boundary * self.L1(pred[boundary_mask], gt_dose[boundary_mask])
                      if boundary_mask.any() else pred.new_tensor(0.0))

        # ── L_Gradient (eq. 6) ────────────────────────────────────────────
        # ‖∇ŷ − ∇y‖₁  where ∇ = 3D Sobel
        p_gx, p_gy, p_gz = self._sobel_gradients(pred)
        g_gx, g_gy, g_gz = self._sobel_gradients(gt_dose)
        grad_l1 = (torch.abs(p_gx - g_gx)
                   + torch.abs(p_gy - g_gy)
                   + torch.abs(p_gz - g_gz)).mean()
        L_gradient = self.w_gradient * grad_l1

        # ── Total (eq. 7) ─────────────────────────────────────────────────
        total = L1_loss + L_ptv + L_oar + L_boundary + L_gradient
        return total


class Loss_AsymmetricPenumbra(nn.Module):
    """
    Asymmetric penumbra loss that directly targets the dose-bleeding failure mode.

    L_Total = L_base + |w_sdw|·L_distweighted + |w_ext|·L_exterior + |w_cov|·L_coverage

    L_base        : L1_mask + w_PTV·L_PTV + w_OAR·L_OAR  (same as Loss_DC_PTV)
    L_distweighted: continuous exp-decay weighted L1, centred on PTV surface
    L_exterior    : relu(pred - gt) outside PTV — penalises over-dose bleeding
    L_coverage    : relu(gt - pred) inside PTV  — penalises under-coverage cold spots
    """

    def __init__(self,
                 n_steps: int = 3,
                 sigma: float = 1.5,
                 w_sdw_init: float = 0.5,
                 w_ext_init: float = 1.0,
                 w_cov_init: float = 0.5):
        super().__init__()
        self.n_steps = n_steps
        self.sigma = sigma

        self.PTV_weight = nn.Parameter(torch.tensor(1.0))
        self.OAR_weight = nn.Parameter(torch.tensor(1.0))
        self.w_sdw = nn.Parameter(torch.tensor(w_sdw_init))
        self.w_ext = nn.Parameter(torch.tensor(w_ext_init))
        self.w_cov = nn.Parameter(torch.tensor(w_cov_init))

        self.L1 = nn.L1Loss(reduction='mean')

        # 3x3x3 all-ones kernel for morphological erosion
        self.register_buffer('erosion_kernel', torch.ones(1, 1, 3, 3, 3))

    def _erode(self, mask: torch.Tensor) -> torch.Tensor:
        conv = F.conv3d(mask, self.erosion_kernel, padding=1)
        return (conv >= 27.0 - 1e-3).float()

    def _dilate(self, mask: torch.Tensor) -> torch.Tensor:
        return (F.max_pool3d(mask, kernel_size=3, stride=1, padding=1) > 0.5).float()

    def _build_distance_weight_map(self, ptv_binary: torch.Tensor) -> torch.Tensor:
        # Initialise all voxels to far distance; override as bands are found
        d_map = torch.full_like(ptv_binary, float(self.n_steps + 1))

        # Interior bands via iterative erosion
        prev = ptv_binary
        for k in range(self.n_steps + 1):
            curr = self._erode(prev)
            band = (prev - curr).clamp(min=0)  # ring peeled off at step k
            d_map = torch.where(band > 0.5, torch.full_like(d_map, float(k)), d_map)
            prev = curr
            if not curr.any():
                break

        # Exterior bands via iterative dilation
        prev = ptv_binary
        for k in range(1, self.n_steps + 1):
            curr = self._dilate(prev)
            band = (curr - prev).clamp(min=0)  # new ring added outside
            d_map = torch.where(band > 0.5, torch.full_like(d_map, float(k)), d_map)
            prev = curr

        return torch.exp(-d_map / self.sigma)

    def forward(self, pred, gt, PTVs, OAR):
        gt_dose = gt[0]
        possible_dose_mask = gt[1]

        # ── L_base (mirrors Loss_DC_PTV) ─────────────────────────────────
        pred_m = pred[possible_dose_mask > 0]
        gt_m   = gt_dose[possible_dose_mask > 0]
        L1_loss = self.L1(pred_m, gt_m)

        ptv_mask = PTVs > 0
        L_ptv = (self.PTV_weight * self.L1(pred[ptv_mask], gt_dose[ptv_mask])
                 if ptv_mask.any() else pred.new_tensor(0.0))

        oar_mask = OAR.sum(dim=1, keepdim=True) > 0
        L_oar = (self.OAR_weight * self.L1(pred[oar_mask], gt_dose[oar_mask])
                 if oar_mask.any() else pred.new_tensor(0.0))

        L_base = L1_loss + L_ptv + L_oar

        # ── L_distweighted: exponential-decay surface-centred L1 ─────────
        ptv_binary = (PTVs > 0).float()
        w_dist = self._build_distance_weight_map(ptv_binary)   # [B,1,D,H,W]
        dose_float = (possible_dose_mask > 0).float()
        n_vox = dose_float.sum().clamp(min=1)
        L_distweighted = (w_dist * torch.abs(pred - gt_dose) * dose_float).sum() / n_vox

        # ── L_exterior: asymmetric over-dose penalty outside PTV ─────────
        outside_mask = (possible_dose_mask > 0) & ~ptv_mask
        if outside_mask.any():
            L_exterior = F.relu(pred[outside_mask] - gt_dose[outside_mask]).mean()
        else:
            L_exterior = pred.new_tensor(0.0)

        # ── L_coverage: asymmetric under-dose penalty inside PTV ─────────
        if ptv_mask.any():
            L_coverage = F.relu(gt_dose[ptv_mask] - pred[ptv_mask]).mean()
        else:
            L_coverage = pred.new_tensor(0.0)

        total = (L_base
                 + self.w_sdw.abs() * L_distweighted
                 + self.w_ext.abs() * L_exterior
                 + self.w_cov.abs() * L_coverage)
        return total