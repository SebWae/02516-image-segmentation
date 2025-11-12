# this script is used to run the actual experiments
import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms as T

from lib.dataset import DriveDataset, PH2Dataset
from lib.model import EncDec, UNet
from lib.train import train_model


# hyperparameters
batch_size = 4
dataset_name = "DRIVE"
factor = 0.3
gamma = 2.0
img_size = 256
k_folds = 8
loss_func = "cross_entropy"
lr = 1e-4
model_type = "EncDec"
patience_scheduler = 5
patience_train = 20
seed = 42
train_prop = 0.8
weight_decay = 1e-5

# data augmentation
transform = T.Compose([
    #T.Resize(img_size),                 # resize shorter side to img_size
    T.CenterCrop((img_size, img_size)), # crop to square
    T.ToTensor()
])

hyperparam_vals = {"batch_size": batch_size, 
                   "dataset": dataset_name,
                   "factor": factor,
                   "gamma": gamma,
                   "img_size": img_size,
                   "k_folds": k_folds, 
                   "loss_func": loss_func,
                   "lr": lr,
                   "model_type": model_type,
                   "patience_scheduler": patience_scheduler,
                   "patience_train": patience_train,
                   "seed": seed, 
                   "train_prop": train_prop, 
                   "weight_decay": weight_decay,
                   }

# printing hyperparameter values
for param, val in hyperparam_vals.items():
    print(f"{param}: {val}")

# defining dataset
dataset_dict = {"DRIVE": DriveDataset,
                "PH2": PH2Dataset,
                }
dataset = dataset_dict[dataset_name](transform=transform)

# dictionary to find model
model_dict = {
            "EncDec": EncDec,
            "UNet": UNet,
              }

# randomly divide the dataset into a train and test set
train_size = int(train_prop * len(dataset))
test_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(seed)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

# initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# apply k-fold cross-validation
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
fold_results = {"train_losses": [],
                "train_dice_scores": [],
                "val_losses": [],
                "val_dice_scores": [],
                }

for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
    print(f"\n--- Fold {fold + 1}/{k_folds} ---")
    # initialize model
    model = model_dict[model_type]()
    model.to(device)

    # initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=factor, patience=patience_scheduler
    )

    # splits the train_dataset into a train and validation sampler
    train_subsampler = Subset(train_dataset, train_idx)
    val_subsampler = Subset(train_dataset, val_idx)

    # initialize dataloaders
    train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

    # train and validate model on fold
    out_dict = train_model(model=model,
                           train_loader=train_loader,
                           val_loader=val_loader,
                           device=device,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           loss_func=loss_func,
                           gamma=gamma,
                           patience=patience_train,
                           save_model=False,
                           )

    # performance metrics after final epoch
    train_loss = out_dict["train_loss"][-2]
    train_dice_score = out_dict["train_dice"][-2]
    val_loss = out_dict["val_loss"][-2]
    val_dice_score = out_dict["val_dice"][-2]

    # appending to fold_results 
    fold_results["train_losses"].append(train_loss)
    fold_results["train_dice_scores"].append(train_dice_score)
    fold_results["val_losses"].append(val_loss)
    fold_results["val_dice_scores"].append(val_dice_score)

# report performance across all folds 
print("\n--- Performance across all folds ---")
for metric, vals in fold_results.items():
    avg_val = np.mean(vals)
    print(f"Avg. {metric}: {avg_val:.4f}")
