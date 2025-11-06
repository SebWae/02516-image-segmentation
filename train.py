from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset

# this script should include a train function taking one of the models as input

# applies 8-fold cross-validation for the PH2 dataset and 4-fold cross-validation for the DRIVE dataset
# each fold should act the role as the validation set
# performance is reported as the average across all folds
# applies early-stopping when validation loss increases (after x initial epochs) or only decreases by something smaller than a tolerance (e.g. e-4)


def train(k_folds, train_dataset, n_epochs=1000, random_state=42) -> dict:
    out_dict = {"train_loss": [],
                "train_dice": [],
                "val_loss": [],
                "val_dice": [],
                "n_epochs": 0,
                }
    prev_val_loss = 1e10

    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
    for epoch in range(n_epochs):
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
            print(f"\n--- Fold {fold + 1}/{k_folds} ---")

            # train and validation samplers for this fold
            train_subsampler = Subset(train_dataset, train_idx)
            val_subsampler = Subset(train_dataset, val_idx)

            # initialize dataloaders
            train_loader = DataLoader(train_subsampler, batch_size=4, shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=4, shuffle=False)


    return out_dict