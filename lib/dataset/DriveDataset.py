# this script should contain a dataset class loading the DRIVE dataset 
# the class should be instantiated and passed to a DataLoader

from glob import glob
from PIL import Image
import torch
from torchvision import transforms as T

class DriveDataset(torch.utils.data.Dataset):
    def __init__(self, 
    root_dir='DRIVE',
    transform=None
):      
        # combining the pre-defined training and test folders
        image_paths = sorted(glob(f'{root_dir}/training/images/*.tif')) + sorted(glob(f'{root_dir}/test/images/*.tif'))
        mask_paths = sorted(glob(f'{root_dir}/training/masks/*.png')) + sorted(glob(f'{root_dir}/test/masks/*.png'))

        # checking that there are equally many images and masks
        assert len(image_paths) == len(mask_paths)
    
        self.image_paths = image_paths
        self.mask_paths = mask_paths
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
            mask = self.transform(mask)
        else:
            image = T.ToTensor()(image)
            mask = T.ToTensor()(mask)

        return image, mask