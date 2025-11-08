import numpy as np
import torch
import torch.nn as nn

from lib.eval_metrics import compute_dice
from lib.losses import FocalLoss, PixelWeightedCrossEntropyLoss

# this script should include a train function taking one of the models as input

# applies 8-fold cross-validation for the PH2 dataset and 4-fold cross-validation for the DRIVE dataset
# each fold should act the role as the validation set
# performance is reported as the average across all folds
# applies early-stopping when validation loss increases (after x initial epochs) or only decreases by something smaller than a tolerance (e.g. e-4)


def train_model(model, 
                train_loader,
                val_loader, 
                device, 
                optimizer, 
                scheduler, 
                loss="cross_entropy", 
                n_epochs=1000, 
                patience=5, 
                tol=1e-4,
                ) -> dict:
    """
    Function to train a segmentation model.

    Parameters:
    - model:            Segmentation model to be trained.
    - train_loader:     Dataloader for training data.
    - val_loader        Dataloader for validation data.
    - device:           Device on which to perform the training (GPU if available else CPU).
    - optimizer:        Optimizer to be using during training, e.g., SGD or Adam. 
    - scheduler:        Scheduler to adjust learning rate during training. 
    - loss:             Loss function to be used, must be either cross-entropy, focal loss, or cross-entropy with positive weights (default is cross-entropy).
    - n_epochs:         Maximum number of epochs to train for (default value 1000).
    - patience:         Number of epochs to be ran before early stopping can be triggered (default value 5). 
    - tol:              Early stopping is applied if improvement in validation loss is lower than the tolerance (default value 0.0001).

    Returns:
    - out_dict:         Dictionary containing the loss and dice score per epoch on the training and validation set, respectively.
    """
    out_dict = {"train_loss": [],
                "train_dice": [],
                "val_loss": [],
                "val_dice": [],
                "n_epochs": 0,
                }
    
    # defining the loss function to be applied
    loss_dict = {"cross_entropy":       nn.CrossEntropyLoss(label_smoothing=0.1),
                 "focal_loss":          FocalLoss(gamma=2.0, alpha=None),
                 "cross_entropy_pw":    PixelWeightedCrossEntropyLoss(),
                 }
    criterion = loss_dict[loss]
    
    prev_val_loss = 1e10

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        train_dice_scores = []
        val_losses = []
        val_dice_scores = []

        for image, mask in train_loader:
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad()

            # forward pass through model (output should have shape batch_size x 1 x H x W or batch_size x H x W)
            output = model(image)

            # computing the loss per video
            loss = criterion(output, mask)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # converting the output to binary masks
            predicted = (output >= 0.5).int()

            # computing the dice score on the training data
            dice = compute_dice(predicted, mask)
            train_dice_scores.append(dice)

        model.eval()
        with torch.no_grad():
            for image, mask in val_loader:
                image, mask = image.to(device), mask.to(device)
                output = model(image)

                val_losses.append(criterion(output, mask).cpu().item())
                predicted = (output >= 0.5).int()
                dice = compute_dice(predicted, mask)
                val_dice_scores.append(dice)

        val_loss = np.mean(val_losses)
        scheduler.step(val_loss)

        # computing loss and dice scores for epoch
        train_loss = np.mean(train_losses)
        train_dice = np.mean(train_dice_scores)
        val_dice = np.mean(val_dice_scores)

        # appending to lists in out_dict
        out_dict['train_loss'].append(train_loss)
        out_dict['train_dice'].append(train_dice)
        out_dict['val_loss'].append(val_loss)
        out_dict['val_dice'].append(val_dice)

        # printing out the results for epoch
        print(f"Train loss: {train_loss:.3f}\t val: {val_loss:.3f}\t",
                f"Train dice score: {train_dice:.4f}%\t val: {val_dice:.4f}%")
        
        # increment the number of epochs 
        out_dict['n_epochs'] += 1

        # check if early stopping should be applied
        if epoch + 1 > patience:
            val_loss_diff = prev_val_loss - val_loss

            if val_loss_diff < 0:
                print(f"Current validation loss ({val_loss}) is larger than for the previous epoch ({prev_val_loss})!")
                print("Early stopping applies!")
                break

            elif val_loss_diff < tol:
                print(f"Validation loss only improved by {val_loss_diff} which is lower than the tolerance of {tol}!")
                print("Early stopping applies!")
                break
        
        # save current model
        else:
            print(f"Validation loss improved from {prev_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            torch.save(model.state_dict(), 'best_model.pt')

        # setting the prev_val_loss to the current val_loss
        prev_val_loss = val_loss
                
    return out_dict