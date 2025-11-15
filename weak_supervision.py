from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms as T

from lib.dataset import PH2WeakDataset
from lib.model import EncDec
# from lib.model import UNet   # ← you can switch model here
from lib.train_weak import train_model_weak

# ---- Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_size = 128
batch_size = 16
train_split = 0.8

# Separate transforms for images and masks
image_transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
])

mask_transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
])

# ---- Dataset Paths ----
data_root = Path("/dtu/datasets1/02516/PH2_Dataset_images")

image_paths = []
mask_paths = []

# Collect all images and masks
for patient_folder in sorted(data_root.glob("IMD*")):
    img_file = next(patient_folder.glob("*_Dermoscopic_Image/*.bmp"))
    mask_file = next(patient_folder.glob("*_lesion/*.bmp"))
    image_paths.append(img_file)
    mask_paths.append(mask_file)

# ---- Verify dataset paths ----
if len(image_paths) == 0 or len(mask_paths) == 0:
    raise FileNotFoundError(f"No images or masks found!\nImages: {len(image_paths)}, Masks: {len(mask_paths)}")
print(f"Found {len(image_paths)} images and {len(mask_paths)} masks.")

# ---- Ablation: Different number of clicks ----
for num_clicks in [1, 5, 10, 20, 50]:
    print(f"\n===== Training with {num_clicks} positive/negative clicks =====")

    # ---- Load dataset with separate transforms ----
    dataset = PH2WeakDataset(
        image_paths,
        mask_paths,
        image_transform=image_transform,
        mask_transform=mask_transform,
        num_pos_clicks=num_clicks,
        num_neg_clicks=num_clicks
    )

    if len(dataset) == 0:
        print("Dataset is empty. Skipping this configuration.")
        continue

    # ---- Train/val split ----
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size

    if train_size == 0 or val_size == 0:
        print("Split too small for this dataset size. Skipping this configuration.")
        continue

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ---- Initialize model ----
    model = EncDec().to(device)
    # model = UNet().to(device)  # uncomment to use UNet

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=5
    )

    # ---- Train ----
    train_model_weak(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=10
    )

    print(f"Finished training with {num_clicks} clicks.")
