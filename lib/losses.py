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

