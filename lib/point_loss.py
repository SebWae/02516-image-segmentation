import torch
import torch.nn.functional as F

def point_loss(preds, pos_clicks, neg_clicks):
    """
    Computes BCE loss only on clicked pixel locations.
    preds: (B, 1, H, W)
    pos_clicks, neg_clicks: list of tensors [(N_pos, 2), (N_neg, 2)]
    """
    total_loss = 0.0
    B, _, H, W = preds.shape

    # Convert predictions to probabilities using sigmoid
    preds = torch.sigmoid(preds)

    for i in range(B):
        p = preds[i, 0]  # (H, W) prediction for the i-th image in the batch

        # Positive clicks -> label = 1
        if len(pos_clicks[i]) > 0:
            pos_coords = pos_clicks[i]
            pos_vals = p[pos_coords[:, 0], pos_coords[:, 1]]  # Get prediction values at positive click locations
            total_loss += F.binary_cross_entropy(pos_vals, torch.ones_like(pos_vals, device=p.device))

        # Negative clicks -> label = 0
        if len(neg_clicks[i]) > 0:
            neg_coords = neg_clicks[i]
            neg_vals = p[neg_coords[:, 0], neg_coords[:, 1]]  # Get prediction values at negative click locations
            total_loss += F.binary_cross_entropy(neg_vals, torch.zeros_like(neg_vals, device=p.device))

    return total_loss / B
