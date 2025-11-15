import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T

class PH2WeakDataset(Dataset):
    """
    Weakly supervised dataset for PH2 — simulates user clicks on object and background.
    Each sample returns (image, pos_clicks, neg_clicks, mask).
    """

    def __init__(self, image_paths, mask_paths,
                 image_transform=None,
                 mask_transform=None,
                 num_pos_clicks=10,
                 num_neg_clicks=10):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.num_pos_clicks = num_pos_clicks
        self.num_neg_clicks = num_neg_clicks

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load images
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # grayscale mask

        # Apply separate transforms
        if self.image_transform:
            image = self.image_transform(image)

        if self.mask_transform:
            mask = self.mask_transform(mask)

        # Ensure mask is binary (0 or 1)
        mask = (mask > 0.5).float()

        # Generate weak supervision clicks (positive and negative)
        pos_clicks, neg_clicks = self._generate_clicks(mask)

        return image, pos_clicks, neg_clicks, mask

    def _generate_clicks(self, mask):
        """
        Generate random positive and negative clicks within the mask.
        mask: tensor of shape (H, W)
        """

        mask_np = mask.squeeze().numpy()
        pos_coords = np.argwhere(mask_np == 1)
        neg_coords = np.argwhere(mask_np == 0)

        pos_clicks = torch.empty((0, 2), dtype=torch.long)
        neg_clicks = torch.empty((0, 2), dtype=torch.long)

        if len(pos_coords) > 0:
            idxs = np.random.choice(len(pos_coords),
                                    min(self.num_pos_clicks, len(pos_coords)),
                                    replace=False)
            pos_clicks = torch.tensor(pos_coords[idxs], dtype=torch.long)

        if len(neg_coords) > 0:
            idxs = np.random.choice(len(neg_coords),
                                    min(self.num_neg_clicks, len(neg_coords)),
                                    replace=False)
            neg_clicks = torch.tensor(neg_coords[idxs], dtype=torch.long)

        return pos_clicks, neg_clicks
