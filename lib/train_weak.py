import torch
import torch.nn.functional as F
from .weak_losses import point_loss
from .eval_metrics import compute_dice, iou_score, accuracy_score, sensitivity_score, specificity_score

def train_model_weak(model, train_loader, val_loader, device, optimizer, scheduler, patience=10):
    best_val = float('inf')
    counter = 0

    for epoch in range(1, 51):
        model.train()
        total_train_loss = 0

        # Training loop
        for imgs, pos_clicks, neg_clicks, masks in train_loader:
            imgs = imgs.to(device)
            preds = model(imgs)  # raw logits output

            loss = point_loss(preds, pos_clicks, neg_clicks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss = 0
        dice = 0
        iou = 0
        acc = 0
        sens = 0
        spec = 0

        with torch.no_grad():
            for imgs, pos_clicks, neg_clicks, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)  # NO sigmoid here!

                # Calculate BCE loss with sigmoid inside
                val_loss += F.binary_cross_entropy(torch.sigmoid(preds), masks).item()

                # Compute metrics (they apply sigmoid internally)
                dice += compute_dice(preds, masks)
                iou += iou_score(preds, masks)
                acc += accuracy_score(preds, masks)
                sens += sensitivity_score(preds, masks)
                spec += specificity_score(preds, masks)

        # Average over validation batches
        N = len(val_loader)
        val_loss /= N
        dice /= N
        iou /= N
        acc /= N
        sens /= N
        spec /= N

        print(f"[Epoch {epoch:02d}] Train={total_train_loss:.4f} | Val={val_loss:.4f} | "
              f"Dice={dice:.4f} | IoU={iou:.4f} | Acc={acc:.4f} | Sens={sens:.4f} | Spec={spec:.4f}")

        # Scheduler step with validation loss
        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
