# this script should contain a dataset class loading the DRIVE dataset 
# the class should be instantiated and passed to a DataLoader

from glob import glob
from PIL import Image
import torch
from torchvision import transforms as T

class DriveDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='DRIVE',
    split='train', 
    transform=None
):
        self.image_paths = sorted(glob(f'{root_dir}/{split}/images/*.tif'))
        self.mask_paths = sorted(glob(f'{root_dir}/{split}/masks/*.png'))
        self.split = split
        self.transform = transform
       
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = T.ToTensor()(mask)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)

        return image, mask