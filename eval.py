from collections import defaultdict

import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms as T

from lib.eval_metrics import evaluate_all
from lib.dataset import DriveDataset, PH2Dataset
from lib.losses import FocalLoss
from lib.model import EncDec, UNet


# hyperparameters
batch_size = 4
dataset_name = "DRIVE"
factor = 0.3
gamma = 2.0
w = 2
img_size = 256
k_folds = 8
loss_func = "cross_entropy_pw"
lr = 1e-4
model_type = "EncDec"
n_epochs = 25   # use number of epochs before early stopping was triggered during hyperparameter tuning
patience_scheduler = 5
patience_train = 20
seed = 42
train_prop = 0.8
weight_decay = 1e-5
best_model_name = "best_model"

# data augmentation
transform = T.Compose([
    # T.Resize(img_size),                 # resize shorter side to img_size
    T.CenterCrop((img_size, img_size)), # crop to square
    T.ToTensor()
])

hyperparam_vals = {"batch_size": batch_size, 
                   "dataset": dataset_name,
                   "factor": factor,
                   "gamma": gamma,
                   "w": w,
                   "img_size": img_size,
                   "k_folds": k_folds, 
                   "loss_func": loss_func,
                   "lr": lr,
                   "model_type": model_type,
                   "n_epochs": n_epochs,
                   "patience_scheduler": patience_scheduler,
                   "patience_train": patience_train,
                   "seed": seed, 
                   "train_prop": train_prop, 
                   "weight_decay": weight_decay,
                   "best_model_name": best_model_name,
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
model_dict = {"EncDec": EncDec,
              "UNet": UNet,
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

# train model
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

    # computing average loss and metric on the train set for epoch
    avg_train_loss = np.mean(epoch_results["train_losses"])
    avg_train_dice = np.mean(epoch_results["train_dice_scores"])
    avg_train_iou = np.mean(epoch_results["train_iou_scores"])
    avg_train_acc = np.mean(epoch_results["train_acc_scores"])
    avg_train_sens = np.mean(epoch_results["train_sens_scores"])
    avg_train_spec = np.mean(epoch_results["train_spec_scores"])

    # appending to lists in out_dict
    out_dict['train_loss'].append(avg_train_loss)
    out_dict['train_dice'].append(avg_train_dice)
    out_dict['train_iou'].append(avg_train_iou)
    out_dict['train_acc'].append(avg_train_acc)
    out_dict['train_sens'].append(avg_train_sens)
    out_dict['train_spec'].append(avg_train_spec)

    # printing out the epoch results
    print(f"Epoch {epoch + 1} results:\n",
          f"Results on train:\t loss: {avg_train_loss:.4f}\t dice: {avg_train_dice:.4f}\t iou: {avg_train_iou:.4f}\t acc: {avg_train_acc:.4f}\t sens: {avg_train_iou:.4f}\t spec: {avg_train_acc:.4f}\n",
          )

# save model
# print(f"Saving model as '{best_model_name}.pt'")
# torch.save(model.state_dict(), f'{best_model_name}.pt')

# # loading the saved model
# model.load_state_dict(torch.load('best_model.pt'))
# model.to(device)

# evaluate performance on test set
eval_results = defaultdict(list)

for image, mask in test_loader:
    image, mask = image.to(device), mask.to(device)
    optimizer.zero_grad()

    # forward pass through model (output should have shape batch_size x 1 x H x W or batch_size x H x W)
    output = model(image)

    # computing the loss for the batch
    train_loss = criterion(output, mask)

    # updating model weights by performing a backward step
    train_loss.backward()
    optimizer.step()
    eval_results["test_losses"].append(train_loss.item())

    # converting the predicted and ground truth masks to binary masks
    probs = torch.sigmoid(output)
    predicted = (probs >= 0.5).bool()
    mask = (mask > 0.5).bool()

    # computing evaluation metrics on the training data
    eval_metric_dict_train = evaluate_all(pred_mask=predicted, gt_mask=mask)
    eval_results["test_dice_scores"].append(eval_metric_dict_train["dice"])
    eval_results["test_iou_scores"].append(eval_metric_dict_train["iou"])
    eval_results["test_acc_scores"].append(eval_metric_dict_train["accuracy"])
    eval_results["test_sens_scores"].append(eval_metric_dict_train["sensitivity"])
    eval_results["test_spec_scores"].append(eval_metric_dict_train["specificity"])

# computing average loss and metric on the train set for epoch
avg_test_loss = np.mean(epoch_results["train_losses"])
avg_test_dice = np.mean(epoch_results["test_dice_scores"])
avg_test_iou = np.mean(epoch_results["test_iou_scores"])
avg_test_acc = np.mean(epoch_results["test_acc_scores"])
avg_test_sens = np.mean(epoch_results["test_sens_scores"])
avg_test_spec = np.mean(epoch_results["test_spec_scores"])

# printing out the test results
print(f"Results on test:\t loss: {avg_test_loss:.4f}\t dice: {avg_test_dice:.4f}\t iou: {avg_test_iou:.4f}\t acc: {avg_test_acc:.4f}\t sens: {avg_test_sens:.4f}\t spec: {avg_test_spec:.4f}\n")