# this script is used to run the actual experiments
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, random_split, Subset

from lib.dataset import DriveDataset, PH2Dataset
from lib.model import EncDec
from lib.train import train_model

# initialize dataset, either DriveDataset or PH2Dataset
dataset = PH2Dataset()
dataset_name = dataset.name

# hyperparameters
batch_size = 4
factor = 0.3
k_folds = 8
lr = 1e-4
patience_scheduler = 5
patience_train = 10
seed = 42
train_prop = 0.8
weight_decay = 1e-5

hyperparam_vals = {"batch_size": batch_size, 
                   "dataset": dataset_name,
                   "factor": factor,
                   "k_folds": k_folds, 
                   "lr": lr,
                   "patience_scheduler": patience_scheduler,
                   "patience_train": patience_train,
                   "seed": seed, 
                   "train_prop": train_prop, 
                   "weight_decay": weight_decay,
                   }

# printing hyperparameter values
for param, val in hyperparam_vals.items():
    print(f"{param}: {val}")

# randomly divide the dataset into a train and test set
train_size = int(train_prop * len(dataset))
test_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(seed)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

# initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize model
model = EncDec()
model.to(device)

# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# initialize learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=factor, patience=patience_scheduler
)

# apply k-fold cross-validation
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
fold_results = {"train_losses": [],
                "train_dice_scores": [],
                "val_losses": [],
                "val_dice_scores": [],
                }

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
    print(f"\n--- Fold {fold + 1}/{k_folds} ---")

    train_subsampler = Subset(train_dataset, train_idx)
    val_subsampler = Subset(train_dataset, val_idx)

    # initialize dataloaders
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    # train and validate model on fold
    out_dict = train_model()

    # performance metrics after final epoch
    train_loss = out_dict["train_losses"][-1]
    train_dice_score = out_dict["train_dice_scores"][-1]
    val_loss = out_dict["val_losses"][-1]
    val_dice_score = out_dict["val_dice_scores"][-1]

    # appending to fold_results 
    fold_results["train_losses"].append(train_loss)
    fold_results["train_dice_scores"].append(train_dice_score)
    fold_results["val_losses"].append(val_loss)
    fold_results["val_dice_scores"].append(val_dice_score)

# report performance across all folds 
print("\n--- Performance across all folds ---\n")
for metric, vals in fold_results.items():
    avg_val = np.mean(vals)
    print(f"\nAvg. {metric}: {avg_val:.4f}")
