# this script should include the three loss functions to test
# cross-entropy (already available through torch.nn.CrossEntropyLoss)
# focal loss
# cross-entropy with positive weights

import torch
import torch.nn as nn
import torch.nn.functional as F

# focal loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        """
        alpha: float (weight for class 1 in binary), or tensor/list of per-class weights for multiclass, or None
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        # store alpha; convert later to device as needed
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, (float, int)):
                self.alpha = float(alpha)
            else:
                self.alpha = torch.tensor(alpha, dtype=torch.float)

    def _to_device_alpha(self, device):
        if isinstance(self.alpha, torch.Tensor):
            return self.alpha.to(device)
        return self.alpha

    def forward(self, inputs, targets):
        """
        inputs: logits with shape (N, C, H, W) or (N, 1, H, W) for binary
        targets: (N, H, W) with class indices {0,...,C-1} for multiclass or {0,1} for binary
        """
        if inputs.dim() == 4 and inputs.size(1) == 1:
            # binary focal (use logits)
            logits = inputs.squeeze(1)              # (N, H, W)
            targets_float = targets.float()
            prob = torch.sigmoid(logits)
            p_t = prob * targets_float + (1 - prob) * (1 - targets_float)
            ce_loss = F.binary_cross_entropy_with_logits(logits, targets_float, reduction='none')  # (N,H,W)

            alpha = self._to_device_alpha(inputs.device)
            if alpha is None:
                alpha_factor = 1.0
            elif isinstance(alpha, float):
                # alpha is weight for class 1
                alpha_factor = targets_float * alpha + (1 - targets_float) * (1 - alpha)
            else:
                # alpha tensor, expect length 2
                a = alpha.view(-1)
                if a.numel() == 2:
                    alpha_factor = targets_float * a[1] + (1 - targets_float) * a[0]
                else:
                    # fallback: no alpha
                    alpha_factor = 1.0

            loss = alpha_factor * ((1 - p_t) ** self.gamma) * ce_loss

        else:
            # multiclass focal
            log_prob = F.log_softmax(inputs, dim=1)       # (N, C, H, W)
            prob = torch.exp(log_prob)
            # standard nll per-pixel
            ce_loss = F.nll_loss(log_prob, targets, reduction='none')  # (N,H,W)

            # p_t = prob of true class
            p_t = prob.gather(1, targets.unsqueeze(1)).squeeze(1)      # (N,H,W)

            alpha = self._to_device_alpha(inputs.device)
            if alpha is None:
                alpha_factor = 1.0
            elif isinstance(alpha, float):
                # scalar alpha -> weight for class 1; build per-class alpha only if C==2
                if inputs.size(1) == 2:
                    a0 = 1.0 - alpha
                    a1 = alpha
                    alpha_factor = torch.where(targets == 1, a1, a0).to(inputs.device)
                else:
                    alpha_factor = 1.0
            else:
                # tensor of per-class weights
                alpha_t = alpha.to(inputs.device).view(-1)
                # index per-pixel alpha by class index
                alpha_factor = alpha_t[targets]

            loss = alpha_factor * ((1 - p_t) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# pixel-weighted cross entropy 
class PixelWeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(PixelWeightedCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, weights):
        """
        Args:
            inputs: logits tensor of shape (N, C, H, W)
            targets: ground truth tensor of shape (N, H, W)
            weights: per-pixel weights tensor of shape (N, H, W)
        """
        log_prob = F.log_softmax(inputs, dim=1)
        loss = F.nll_loss(log_prob, targets, reduction='none')
        weighted_loss = (loss * weights).mean()
        return weighted_loss


def compute_class_weights(targets):
    """
    Computes per-pixel weights based on inverse class frequency.
    Args:
        targets: tensor of shape (N, H, W)
    Returns:
        weights: tensor of shape (N, H, W)
    """
    # frequency of each class
    num_fg = (targets == 1).sum().float()
    num_bg = (targets == 0).sum().float()
    total = num_fg + num_bg

    # avoiding division by zero
    w_fg = total / (2 * num_fg + 1e-8)
    w_bg = total / (2 * num_bg + 1e-8)

    # assigning weights
    weights = torch.where(targets == 1, w_fg, w_bg)
    return weights
