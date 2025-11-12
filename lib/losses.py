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
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        if targets.dim() == 4:
            targets = targets.squeeze(1)

        # converting logits to probabilities
        probs = torch.sigmoid(inputs)

        # computing pt (p if mask is 1, 1-p otherwise)
        pt = probs * targets + (1 - probs) * (1 - targets)

        # computing the focal loss of each pixel
        focal_loss = -(1 - pt)**self.gamma * torch.log(pt)

        # computing average focal loss across pixels
        avg_focal_loss = focal_loss.mean()
        
        return avg_focal_loss
 

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
