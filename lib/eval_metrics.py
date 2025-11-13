import torch

def compute_dice(pred_mask, gt_mask) -> float:
    """
    Computes the Dice score between predicted and ground truth masks.
    """
    # Apply threshold (0.5) to predicted mask
    pred_mask = (torch.sigmoid(pred_mask) > 0.5).float()  # Ensure binary
    gt_mask = (gt_mask > 0.5).float()  # Ensure ground truth is binary

    # Convert masks to boolean for bitwise operations
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()

    intersection = (pred_mask & gt_mask).sum(dim=(1, 2))
    total_size = pred_mask.sum(dim=(1, 2)) + gt_mask.sum(dim=(1, 2))
    dice_scores = 2 * intersection / total_size.clamp(min=1)
    return dice_scores.mean().item()


def iou_score(pred_mask, gt_mask) -> float:
    """
    Intersection over Union (IoU) score.
    """
    # Apply threshold (0.5) to predicted mask
    pred_mask = (torch.sigmoid(pred_mask) > 0.5).float()
    gt_mask = (gt_mask > 0.5).float()

    # Convert masks to boolean for bitwise operations
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()

    intersection = (pred_mask & gt_mask).sum(dim=(1, 2))
    union = (pred_mask | gt_mask).sum(dim=(1, 2))
    iou = intersection / union.clamp(min=1)
    return iou.mean().item()


def accuracy_score(pred_mask, gt_mask) -> float:
    """
    Pixel-wise accuracy between prediction and ground truth.
    """
    pred_mask = (torch.sigmoid(pred_mask) > 0.5).float()  # Threshold the predictions
    gt_mask = (gt_mask > 0.5).float()

    # Convert to boolean for accurate comparison
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()

    correct = (pred_mask == gt_mask).float().sum()
    total = torch.numel(pred_mask)
    return (correct / total).item()


def sensitivity_score(pred_mask, gt_mask) -> float:
    """
    Sensitivity (Recall): TP / (TP + FN)
    """
    pred_mask = (torch.sigmoid(pred_mask) > 0.5).float()
    gt_mask = (gt_mask > 0.5).float()

    # Convert to boolean for accurate comparison
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()

    tp = (pred_mask & gt_mask).sum().float()
    fn = (~pred_mask & gt_mask).sum().float()
    return (tp / (tp + fn).clamp(min=1)).item()


def specificity_score(pred_mask, gt_mask) -> float:
    """
    Specificity: TN / (TN + FP)
    """
    pred_mask = (torch.sigmoid(pred_mask) > 0.5).float()
    gt_mask = (gt_mask > 0.5).float()

    # Convert to boolean for accurate comparison
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()

    tn = (~pred_mask & ~gt_mask).sum().float()
    fp = (pred_mask & ~gt_mask).sum().float()
    return (tn / (tn + fp).clamp(min=1)).item()


def evaluate_all(pred_mask, gt_mask) -> dict:
    """
    Computes all metrics and returns a dictionary.
    """
    return {
        "dice": compute_dice(pred_mask, gt_mask),
        "iou": iou_score(pred_mask, gt_mask),
        "accuracy": accuracy_score(pred_mask, gt_mask),
        "sensitivity": sensitivity_score(pred_mask, gt_mask),
        "specificity": specificity_score(pred_mask, gt_mask),
    }
