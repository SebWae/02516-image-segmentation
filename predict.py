# this script is used to run the actual experiments
import torch
from torch.utils.data import DataLoader, random_split, Subset
from lib.dataset import DriveDataset, PH2Dataset

# initialize the dataset, either DriveDataset or PH2Dataset
dataset = PH2Dataset()

# hyperparameters
train_prop = 0.8
seed = 42

# randomly 
train_size = int(train_prop * len(dataset))
test_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(seed)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=generator)

# Step 1: Import data with DataLoader and dataset class


# Step 2: Import model class


# Step 3: Train model


# Step 4: Test model and obtain results

