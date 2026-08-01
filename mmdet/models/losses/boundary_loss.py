import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
from mmdet.registry import MODELS
from mmdet.models.losses.utils import weight_reduce_loss

@MODELS.register_module()
class BoundaryDiceLoss(nn.Module):
    """Boundary-aware Dice Loss for instance segmentation."""
    
    def __init__(self, loss_weight=1.0, kernel_size=3, eps=1e-5, **kwargs):
        super(BoundaryDiceLoss, self).__init__()
        self.loss_weight = loss_weight
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.eps = eps

    def extract_boundary(self, mask):
        """
        使用 MaxPool2d 模拟形态学膨胀和腐蚀来提取边界
        mask: [N, 1, H, W]
        """
        # 膨胀 (Dilation)
        dilation = F.max_pool2d(mask, self.kernel_size, stride=1, padding=self.padding)
        # 腐蚀 (Erosion) - 通过对反转的mask做膨胀来实现
        erosion = 1 - F.max_pool2d(1 - mask, self.kernel_size, stride=1, padding=self.padding)
        # 边界 = 膨胀 - 腐蚀
        boundary = dilation - erosion
        return boundary

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        """
        pred: [N, H, W] or [N, num_points] 模型的预测 logits
        target: [N, H, W] or [N, num_points] 真实的二值 mask
        weight: [N] per-sample loss weights (optional)
        avg_factor: normalization factor (optional)
        """
        # Mask2Former uses point-based dice loss with shape [N, num_points]
        if pred.dim() == 2:
            pred_prob = pred.sigmoid()
            intersection = torch.sum(pred_prob * target, dim=1)
            union = torch.sum(pred_prob, dim=1) + torch.sum(target, dim=1)
            dice_loss = 1 - (2.0 * intersection + self.eps) / (union + self.eps)
            loss = self.loss_weight * weight_reduce_loss(dice_loss, weight, 'mean', avg_factor)
            return loss

        # 统一维度到 [N, 1, H, W]
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1).float()

        # 将 logits 转换为概率
        pred_prob = pred.sigmoid()

        # 提取 GT 的边界
        with torch.no_grad():
            target_boundary = self.extract_boundary(target)

        # 提取预测的边界 (软边界)
        pred_boundary = self.extract_boundary(pred_prob)

        # 计算边界区域的 Dice Loss
        intersection = torch.sum(pred_boundary * target_boundary, dim=(2, 3))
        union = torch.sum(pred_boundary, dim=(2, 3)) + torch.sum(target_boundary, dim=(2, 3))

        # shape: (N, 1) -> (N,)
        boundary_dice_loss = (1 - (2.0 * intersection + self.eps) / (union + self.eps)).squeeze(1)

        loss = self.loss_weight * weight_reduce_loss(boundary_dice_loss, weight, 'mean', avg_factor)
        return loss


@MODELS.register_module()
class NonOverlapLoss(nn.Module):
    """Penalize overlap among matched positive instance masks.

    Args:
        loss_weight (float): Loss weight.
        mode (str): ``sum_excess`` penalizes pixels where the summed
            probabilities of positives exceed one. ``pairwise_product``
            penalizes pairwise probability products.
        power (float): Exponent for ``sum_excess``.
        eps (float): Numerical stability constant.
    """

    def __init__(self,
                 loss_weight=1.0,
                 mode='sum_excess',
                 power=2.0,
                 eps=1e-6,
                 **kwargs):
        super().__init__()
        assert mode in ('sum_excess', 'pairwise_product')
        self.loss_weight = loss_weight
        self.mode = mode
        self.power = power
        self.eps = eps

    def forward(self,
                pred,
                group_sizes,
                group_weights=None,
                avg_factor=None,
                reduction_override=None):
        """Compute non-overlap loss.

        Args:
            pred (Tensor): Matched positive mask logits, shape (N, H, W).
            group_sizes (Sequence[int]): Number of positives for each image.
            group_weights (Tensor, optional): Per-image scalar weights.
            avg_factor (float, optional): Kept for config API compatibility.
        """
        if pred.numel() == 0:
            return pred.sum() * self.loss_weight

        probs = pred.sigmoid()
        losses = []
        start = 0
        valid_group_idx = 0
        for group_size in group_sizes:
            end = start + int(group_size)
            group = probs[start:end]
            start = end
            if group.shape[0] <= 1:
                continue

            if self.mode == 'sum_excess':
                excess = torch.relu(group.sum(dim=0) - 1.0)
                loss = excess.pow(self.power).mean()
            else:
                flat = group.flatten(1)
                pairwise = torch.matmul(flat, flat.t())
                areas = flat.sum(dim=1)
                norm = areas[:, None] + areas[None, :] + self.eps
                pairwise = pairwise / norm
                upper = torch.triu(
                    torch.ones_like(pairwise, dtype=torch.bool), diagonal=1)
                loss = pairwise[upper].mean()

            if group_weights is not None:
                loss = loss * group_weights[valid_group_idx]
            losses.append(loss)
            valid_group_idx += 1

        if not losses:
            return pred.sum() * 0.0

        return self.loss_weight * torch.stack(losses).mean()


@MODELS.register_module()
class KervadecBoundaryLoss(nn.Module):
    """Surface/boundary loss from Kervadec et al., "Boundary loss for
    highly unbalanced segmentation" (MIDL 2019 / MedIA 2021 — the paper's
    arXiv preprint was originally titled "Surface loss for highly
    unbalanced segmentation", hence the alternate name used in
    ``docs/plans.md``).

    Unlike region losses (Dice/CE) which only look at pixel-wise overlap,
    this loss multiplies the predicted foreground probability by a signed
    Euclidean distance map computed from the GT mask: positive outside the
    GT object, negative inside it, zero on the GT contour. Minimizing
    ``E[p(x) * phi(x)]`` therefore directly pulls probability mass away
    from far-outside pixels and towards far-inside pixels, i.e. it
    optimises the boundary location instead of the overlap area — useful
    as a complement to Dice/CE for high-precision boundary alignment.

    The distance transform is computed on CPU with
    ``scipy.ndimage.distance_transform_edt`` under ``torch.no_grad()`` (it
    only ever runs on the GT mask), then moved back to the prediction's
    device. This is only tractable at modest resolution (see
    ``surface_max_res`` in ``Mask2FormerHeadV2``), which is why the caller
    downsamples both pred/target before calling this loss.

    Args:
        loss_weight (float): Loss weight. NOTE: this loss is known to be
            unstable early in training (when predictions are far from GT,
            gradients scale with the — possibly large — distance map).
            The original paper anneals the boundary-loss weight up from a
            small value over training while decreasing the Dice weight;
            ``mmdet.engine.hooks.LossWeightRampHook`` can be attached via
            ``custom_hooks`` to reproduce that schedule by mutating this
            module's ``loss_weight`` in place.
        max_distance (float, optional): Clip the (unsigned) distance map at
            this many pixels before signing, to bound gradient magnitude
            for images with large flat backgrounds. ``None`` disables
            clipping.
        eps (float): Numerical stability constant (unused directly, kept
            for config-interface consistency with the other losses here).
    """

    def __init__(self,
                 loss_weight=1.0,
                 max_distance=None,
                 eps=1e-6,
                 **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.max_distance = max_distance
        self.eps = eps

    def _signed_distance_map(self, target: torch.Tensor) -> torch.Tensor:
        """target: (N, H, W) binary tensor -> (N, H, W) signed distance map
        (positive outside, negative inside, in pixel units), computed on
        CPU with scipy and returned on ``target.device``."""
        target_np = target.detach().cpu().numpy().astype(np.uint8)
        maps = np.empty(target_np.shape, dtype=np.float32)
        for i, m in enumerate(target_np):
            if m.any() and not m.all():
                dist_out = distance_transform_edt(1 - m)
                dist_in = distance_transform_edt(m)
                signed = dist_out - dist_in
            elif m.all():
                # whole crop is foreground: push probability up everywhere.
                signed = -distance_transform_edt(m)
            else:
                # empty mask: no boundary to speak of, contributes ~0 loss
                # once the (near-zero) predicted probability is folded in.
                signed = np.zeros_like(m, dtype=np.float32)
            if self.max_distance is not None:
                signed = np.clip(signed, -self.max_distance, self.max_distance)
            maps[i] = signed
        return torch.from_numpy(maps).to(
            device=target.device, dtype=torch.float32)

    def forward(self, pred, target, weight=None, avg_factor=None,
               reduction_override=None):
        """
        pred: (N, H, W) logits.
        target: (N, H, W) binary GT mask.
        weight: (N,) per-sample loss weights (optional).
        avg_factor: normalization factor (optional).
        """
        with torch.no_grad():
            dist_map = self._signed_distance_map(target)

        probs = pred.sigmoid()
        loss_per_sample = (probs * dist_map).mean(dim=(1, 2))
        loss = self.loss_weight * weight_reduce_loss(
            loss_per_sample, weight, 'mean', avg_factor)
        return loss


@MODELS.register_module()
class CurvatureLoss(nn.Module):
    """Conformal curvature regularization (Track 3 / E5 in
    ``docs/plans.md``) that discourages jagged/wiggly predicted mask
    boundaries while explicitly protecting sharp GT corners (e.g. the ~90°
    corners typical of field boundaries).

    The mean curvature of the predicted probability map's level sets is
    estimated as ``kappa = div(grad(p) / |grad(p)|)`` via Sobel finite
    differences, and penalized with a hinge so that curvature below ``tau``
    is free (this tolerance is what protects genuine corners from being
    smoothed away). The penalty is restricted to a thin band around the GT
    boundary (same dilation/erosion trick as :class:`BoundaryDiceLoss`) so
    flat interior/exterior regions (where kappa is ~0 anyway, but noisy)
    don't dominate the average.

    Args:
        loss_weight (float): Loss weight.
        tau (float): Curvature magnitude below which no penalty is applied
            (protects corners). Curvature here is in units of
            (probability change per pixel) / pixel, so this is expected to
            be a small value; tune per resolution.
        band_kernel_size (int): Kernel size for the boundary-band
            dilation/erosion (same semantics as ``BoundaryDiceLoss``'s
            ``kernel_size``).
        eps (float): Numerical stability constant for gradient
            normalization and the band ratio.
    """

    def __init__(self,
                 loss_weight=1.0,
                 tau=0.3,
                 band_kernel_size=5,
                 eps=1e-6,
                 **kwargs):
        super().__init__()
        self.loss_weight = loss_weight
        self.tau = tau
        self.band_kernel_size = band_kernel_size
        self.band_padding = band_kernel_size // 2
        self.eps = eps

        sobel_x = torch.tensor(
            [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3).contiguous()
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def _grad(self, x: torch.Tensor):
        """x: (N, 1, H, W) -> (gx, gy), each (N, 1, H, W)."""
        gx = F.conv2d(x, self.sobel_x.to(x), padding=1)
        gy = F.conv2d(x, self.sobel_y.to(x), padding=1)
        return gx, gy

    def _boundary_band(self, mask: torch.Tensor) -> torch.Tensor:
        """mask: (N, 1, H, W) binary -> (N, 1, H, W) band-membership in
        {0, 1}, via dilation - erosion (same trick as BoundaryDiceLoss)."""
        dilation = F.max_pool2d(
            mask, self.band_kernel_size, stride=1, padding=self.band_padding)
        erosion = 1 - F.max_pool2d(
            1 - mask, self.band_kernel_size, stride=1,
            padding=self.band_padding)
        return (dilation - erosion).clamp(min=0.0, max=1.0)

    def forward(self, pred, target, weight=None, avg_factor=None,
               reduction_override=None):
        """
        pred: (N, H, W) logits.
        target: (N, H, W) binary GT mask (only used to derive the boundary
            band the penalty is restricted to).
        weight: (N,) per-sample loss weights (optional).
        avg_factor: normalization factor (optional).
        """
        p = pred.unsqueeze(1).sigmoid()  # (N, 1, H, W)

        gx, gy = self._grad(p)
        norm = torch.sqrt(gx * gx + gy * gy + self.eps)
        nx, ny = gx / norm, gy / norm
        nxx, _ = self._grad(nx)
        _, nyy = self._grad(ny)
        kappa = nxx + nyy  # discrete divergence of the unit normal field

        with torch.no_grad():
            band = self._boundary_band(target.unsqueeze(1).float())

        hinge = F.relu(kappa.abs() - self.tau).pow(2) * band
        loss_per_sample = hinge.sum(dim=(1, 2, 3)) / (
            band.sum(dim=(1, 2, 3)) + self.eps)

        loss = self.loss_weight * weight_reduce_loss(
            loss_per_sample, weight, 'mean', avg_factor)
        return loss