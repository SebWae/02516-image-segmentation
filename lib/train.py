from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from lib.eval_metrics import evaluate_all
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
    - loss_func:        Loss function to be used, must be either cross_entropy, focal_loss, or cross_entropy_pw (positive weights). Default is cross_entropy.
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
    out_dict = defaultdict(list)
    
    # converting the weight w to a pytorch tensor for cross_entropy_pw
    w = torch.tensor([w]).to(device)

    # defining the loss function to be applied
    loss_dict = {"cross_entropy":       nn.BCEWithLogitsLoss(),
                 "focal_loss":          FocalLoss(gamma=gamma, alpha=None),
                 "cross_entropy_pw":    nn.BCEWithLogitsLoss(pos_weight=w),
                 }
    criterion = loss_dict[loss_func]
    prev_val_loss = 1e10

    for epoch in range(n_epochs):
        epoch_results = defaultdict(list)
        model.train()

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
            epoch_results["train_losses"].append(train_loss.item())

            # converting the predicted and ground truth masks to binary masks
            probs = torch.sigmoid(output)
            predicted = (probs >= 0.5).bool()
            mask = (mask > 0.5).bool()

            # computing evaluation metrics on the training data
            eval_metric_dict_train = evaluate_all(pred_mask=predicted, gt_mask=mask)
            epoch_results["train_dice_scores"].append(eval_metric_dict_train["dice"])
            epoch_results["train_iou_scores"].append(eval_metric_dict_train["iou"])
            epoch_results["train_acc_scores"].append(eval_metric_dict_train["accuracy"])
            epoch_results["train_sens_scores"].append(eval_metric_dict_train["sensitivity"])
            epoch_results["train_spec_scores"].append(eval_metric_dict_train["specificity"])

        model.eval()
        with torch.no_grad():
            for image, mask in val_loader:
                image, mask = image.to(device), mask.to(device)
                output = model(image)
                val_loss = criterion(output, mask).cpu().item()
                epoch_results["val_losses"].append(val_loss)
                probs = torch.sigmoid(output)
                predicted = (probs >= 0.5).bool()
                mask = (mask > 0.5).bool()
                eval_metric_dict_val = evaluate_all(pred_mask=predicted, gt_mask=mask)
                epoch_results["val_dice_scores"].append(eval_metric_dict_val["dice"])
                epoch_results["val_iou_scores"].append(eval_metric_dict_val["iou"])
                epoch_results["val_acc_scores"].append(eval_metric_dict_val["accuracy"])
                epoch_results["val_sens_scores"].append(eval_metric_dict_val["sensitivity"])
                epoch_results["val_spec_scores"].append(eval_metric_dict_val["specificity"])

        avg_val_loss = np.mean(epoch_results["val_losses"])
        scheduler.step(avg_val_loss)

        # computing average loss and metric on the train set for epoch
        avg_train_loss = np.mean(epoch_results["train_losses"])
        avg_train_dice = np.mean(epoch_results["train_dice_scores"])
        avg_train_iou = np.mean(epoch_results["train_iou_scores"])
        avg_train_acc = np.mean(epoch_results["train_acc_scores"])
        avg_train_sens = np.mean(epoch_results["train_sens_scores"])
        avg_train_spec = np.mean(epoch_results["train_spec_scores"])

        # computing average loss and metric on the train set for epoch
        avg_val_dice = np.mean(epoch_results["val_dice_scores"])
        avg_val_iou = np.mean(epoch_results["val_iou_scores"])
        avg_val_acc = np.mean(epoch_results["val_acc_scores"])
        avg_val_sens = np.mean(epoch_results["val_sens_scores"])
        avg_val_spec = np.mean(epoch_results["val_spec_scores"])

        # appending to lists in out_dict
        out_dict['train_loss'].append(avg_train_loss)
        out_dict['train_dice'].append(avg_train_dice)
        out_dict['train_iou'].append(avg_train_iou)
        out_dict['train_acc'].append(avg_train_acc)
        out_dict['train_sens'].append(avg_train_sens)
        out_dict['train_spec'].append(avg_train_spec)
        out_dict['val_loss'].append(avg_val_loss)
        out_dict['val_dice'].append(avg_val_dice)
        out_dict['val_iou'].append(avg_val_iou)
        out_dict['val_acc'].append(avg_val_acc)
        out_dict['val_sens'].append(avg_val_sens)
        out_dict['val_spec'].append(avg_val_spec)

        # printing out the epoch results
        print(f"Epoch {epoch + 1} results:\n",
              f"Results on train:\t loss: {avg_train_loss:.4f}\t dice: {avg_train_dice:.4f}\t iou: {avg_train_iou:.4f}\t acc: {avg_train_acc:.4f}\t sens: {avg_train_iou:.4f}\t spec: {avg_train_acc:.4f}\n",
              f"Results on val:\t loss: {avg_val_loss:.4f}\t dice: {avg_val_dice:.4f}\t iou: {avg_val_iou:.4f}\t acc: {avg_val_acc:.4f}\t sens: {avg_val_iou:.4f}\t spec: {avg_val_acc:.4f}\n",
              )
        
        # increment the number of epochs 
        out_dict['n_epochs'].append(1)

        # check if early stopping should be applied
        if epoch + 1 > patience:
            val_loss_diff = prev_val_loss - avg_val_loss

            if val_loss_diff < 0:
                print(f"Current validation loss ({avg_val_loss}) is larger than for the previous epoch ({prev_val_loss})!")
                print(f"Trained for {np.sum(out_dict['n_epochs'])} epochs.")
                print("Early stopping applies!")
                break

            elif val_loss_diff < tol:
                print(f"Validation loss only improved by {val_loss_diff} which is lower than the tolerance of {tol}!")
                print(f"Trained for {np.sum(out_dict['n_epochs'])} epochs.")
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