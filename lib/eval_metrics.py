# this script should contain the five evaluation metrics as a function
# dice score
# intersection over union
# accuracy
# sensitivity
# specificity

# and an additional eval function that includes all the individual evaluation metrics
def compute_dice(pred_mask, gt_mask) -> float:
    """
    Computes the dice score of the predicted and ground true mask. 

    Parameters:
    - pred_mask:        Batch of predicted masks (3D tensor).
    - gt_mask:          Batch of ground truth masks (3D tensor).

    Returns: 
    -  avg_dice_score:  The computed dice score (average across masks in batch).
    """
    # ensure channel dimension is removed
    if pred_mask.dim() == 4:
        pred_mask = pred_mask.squeeze(1)
    if gt_mask.dim() == 4:
        gt_mask = gt_mask.squeeze(1)

    # make boolean
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    
    # computing intersection (pixels where both masks are 1)
    intersection = (pred_mask & gt_mask).sum(dim=(1,2))

    # computing the sizes of the masks (pixels with value 1)
    preds_sizes = pred_mask.sum(dim=(1,2))
    target_sizes = gt_mask.sum(dim=(1,2))

    # adding the mask sizes (|A|+|B|)
    total_size = preds_sizes + target_sizes

    # computing the dice score per mask in the batch
    dice_scores = 2 * intersection / total_size

    # average dice score across the batch
    avg_dice_score = dice_scores.mean().item()

    return avg_dice_score
