import numpy as np
import torch
import torch.nn as nn

from lib.eval_metrics import compute_dice
from lib.losses import FocalLoss

# this script should include a train function taking one of the models as input
# applies early-stopping when validation loss increases (after x initial epochs) or only decreases by something smaller than a tolerance (e.g. e-4)


def train_model(model, 
                train_loader,
                val_loader, 
                device, 
                optimizer, 
                scheduler, 
                loss_func="cross_entropy", 
                gamma=2.0,
                w=2,
                n_epochs=1000, 
                patience=5, 
                tol=1e-4,
                save_model=True,
                best_model_name="best_model",
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
    - loss_func:        Loss function to be used, must be either cross-entropy, focal loss, or cross-entropy with positive weights (default is cross-entropy).
    - gamma:            Gamma hyperparameter for focal loss. 
    - w:                Weight parameter for the cross entropy with positive weights. 
    - n_epochs:         Maximum number of epochs to train for (default value 1000).
    - patience:         Number of epochs to be ran before early stopping can be triggered (default value 5). 
    - tol:              Early stopping is applied if improvement in validation loss is lower than the tolerance (default value 0.0001).
    - save_model:       Boolean deciding whether to save the best performing model before early stopping applies (True by default).
    - best_model_name:  Name of the model artefact (default "best_model").

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
    loss_dict = {"cross_entropy":       nn.BCEWithLogitsLoss(),
                 "focal_loss":          FocalLoss(gamma=gamma, alpha=None),
                 "cross_entropy_pw":    nn.BCEWithLogitsLoss(pos_weight=w),
                 }
    criterion = loss_dict[loss_func]
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

            # computing the loss for the batch
            train_loss = criterion(output, mask)

            # updating model weights by performing a backward step
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

            # converting the predicted and ground truth masks to binary masks
            probs = torch.sigmoid(output)
            predicted = (probs >= 0.5).bool()
            mask = (mask > 0.5).bool()

            # computing the dice score on the training data
            train_dice = compute_dice(predicted, mask)
            train_dice_scores.append(train_dice)

        model.eval()
        with torch.no_grad():
            for image, mask in val_loader:
                image, mask = image.to(device), mask.to(device)
                output = model(image)
                val_loss = criterion(output, mask).cpu().item()
                val_losses.append(val_loss)
                probs = torch.sigmoid(output)
                predicted = (probs >= 0.5).bool()
                mask = (mask > 0.5).bool()
                val_dice = compute_dice(predicted, mask)
                val_dice_scores.append(val_dice)

        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)

        # computing loss and dice scores for epoch
        avg_train_loss = np.mean(train_losses)
        avg_train_dice = np.mean(train_dice_scores)
        avg_val_dice = np.mean(val_dice_scores)

        # appending to lists in out_dict
        out_dict['train_loss'].append(avg_train_loss)
        out_dict['train_dice'].append(avg_train_dice)
        out_dict['val_loss'].append(avg_val_loss)
        out_dict['val_dice'].append(avg_val_dice)

        # printing out the epoch results
        print(f"Epoch {epoch + 1} results:\n",
              f"Train loss: {avg_train_loss:.4f}\t Val loss: {avg_val_loss:.4f}\n",
              f"Train dice score: {avg_train_dice:.4f}\t Val dice score: {avg_val_dice:.4f}\n")
        
        # increment the number of epochs 
        out_dict['n_epochs'] += 1

        # check if early stopping should be applied
        if epoch + 1 > patience:
            val_loss_diff = prev_val_loss - avg_val_loss

            if val_loss_diff < 0:
                print(f"Current validation loss ({avg_val_loss}) is larger than for the previous epoch ({prev_val_loss})!")
                print("Early stopping applies!")
                break

            elif val_loss_diff < tol:
                print(f"Validation loss only improved by {val_loss_diff} which is lower than the tolerance of {tol}!")
                print("Early stopping applies!")
                break
        
            else:
                print(f"Validation loss improved from {prev_val_loss:.4f} to {avg_val_loss:.4f}.")

        # save current model
        if save_model:
            print(f"Saving model as '{best_model_name}.pt'")
            torch.save(model.state_dict(), f'{best_model_name}.pt')

        # setting the prev_val_loss to the current val_loss
        prev_val_loss = avg_val_loss
                
    return out_dict