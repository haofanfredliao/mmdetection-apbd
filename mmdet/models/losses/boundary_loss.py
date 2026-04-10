import torch
import torch.nn as nn
import torch.nn.functional as F
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