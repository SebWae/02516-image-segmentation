from sklearn.model_selection import KFold
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from lib.losses import FocalLoss, PixelWeightedCrossEntropyLoss

# this script should include a train function taking one of the models as input

# applies 8-fold cross-validation for the PH2 dataset and 4-fold cross-validation for the DRIVE dataset
# each fold should act the role as the validation set
# performance is reported as the average across all folds
# applies early-stopping when validation loss increases (after x initial epochs) or only decreases by something smaller than a tolerance (e.g. e-4)


def train_model(model, 
                train_dataset, 
                device, 
                optimizer, 
                scheduler, 
                loss="cross_entropy", 
                k_folds=8, 
                n_epochs=1000, 
                batch_size=4, 
                patience=5, 
                random_state=42) -> dict:
    """
    Function to train a segmentation model.

    Parameters:
    - model:            Segmentation model to be trained.
    - train_dataset:    The training set to apply k-fold cross-validation on.
    - device:           Device on which to perform the training (GPU if available else CPU).
    - optimizer:        Optimizer to be using during training, e.g., SGD or Adam. 
    - scheduler:        Scheduler to adjust learning rate during training. 
    - loss:             Loss function to be used, must be either cross-entropy, focal loss, or cross-entropy with positive weights (default is cross-entropy).
    - k_folds:          Number of folds for k-fold cross-validation (default value 8).
    - n_epochs:         Maximum number of epochs to train for (default value 1000).
    - batch_size:       Size of batches when loading the train and validation data from the dataloaders. 
    - patience:         Number of epochs to be ran before early stopping can be triggered (default value 5). 
    - random_state:     Seed for reproducibility purposes (default value 42). 

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

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    for epoch in range(n_epochs):
        model.train()
        train_loss = []
        train_dice = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            print(f"\n--- Fold {fold + 1}/{k_folds} ---")

            # train and validation samplers for this fold
            train_subsampler = Subset(train_dataset, train_idx)
            val_subsampler = Subset(train_dataset, val_idx)

            # initialize dataloaders
            train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

            for image, mask in train_loader:
                image, mask = image.to(device), mask.to(device)
                optimizer.zero_grad()

                # forward pass through model (output should have shape batch_size x 1 x H x W or batch_size x H x W)
                output = model(image)

                # computing the loss per video
                loss = criterion(output, mask)
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

                # converting the output to binary masks
                predicted = (output >= 0.5).float()

                # computing the dice score on the training data
                dice = compute_dice(predicted, mask)
                train_dice.append(dice)

                
    return out_dict