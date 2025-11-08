import numpy as np
from sklearn.model_selection import KFold
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms as T

from lib.dataset import DriveDataset, PH2Dataset
from lib.model import EncDec
from lib.train import train_model


# hyperparameters
batch_size = 4
factor = 0.3
gamma = 2.0
img_size = 128
k_folds = 8
label_smoothing = 0.1
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
    T.Resize(img_size),                 # resize shorter side to img_size
    T.CenterCrop((img_size, img_size)), # crop to square
    T.ToTensor()
])

# initialize dataset, either DriveDataset or PH2Dataset
dataset = PH2Dataset(transform=transform)
dataset_name = dataset.name

hyperparam_vals = {"batch_size": batch_size, 
                   "dataset": dataset_name,
                   "factor": factor,
                   "gamma": gamma,
                   "img_size": img_size,
                   "k_folds": k_folds, 
                   "label_smoothing": label_smoothing,
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

# dictionary to find model
model_dict = {"EncDec": EncDec,
              }

# randomly divide the dataset into a train and test set
train_size = int(train_prop * len(dataset))
test_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(seed)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

# initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initialize model
model = model_dict[model_type]()
model.to(device)

# initialize optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# initialize learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=factor, patience=patience_scheduler
)

# initialize dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# train and validate model on fold
out_dict = train_model(model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        loss_func=loss_func,
                        label_smoothing=label_smoothing,
                        gamma=gamma,
                        patience=patience_train,
                        save_model=False,
                        )

# evaluate performance on test set