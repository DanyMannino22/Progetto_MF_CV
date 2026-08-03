"""
dataset_metric.py

Dataset per il training contrastive (instance discrimination) del modello
di metric learning.

Per ogni oggetto estratto, genera DUE view augmentate indipendenti. Le
augmentation sono scelte per simulare le trasformazioni REALISTICHE che
un oggetto subisce quando viene copiato/riutilizzato in una manipolazione:
    - resize/crop (scala diversa)
    - compressione JPEG (quasi sempre presente dopo un salvataggio)
    - variazioni di colore/luminosita' (per amalgamarsi alla nuova scena)
    - flip orizzontale, piccola rotazione

Uso rapido (test standalone):
    python dataset_metric.py
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# PARAMETRI DI CONFIGURAZIONE (modifica qui se serve)
# ============================================================
EXTRACTED_OBJECTS_DIR = "data/extracted_objects"   # crop estratti da GT (dataset "pulito")
IMAGE_SIZE = 224                                    # risoluzione input per ResNet18
SPLIT_TO_TEST = "train"                             # split usato dal sanity check in fondo al file


def get_contrastive_transforms(image_size: int):
    """Augmentation 'realistiche', pensate per simulare le trasformazioni
    che un oggetto manipolato subisce quando viene riutilizzato."""
    return A.Compose([
        A.RandomResizedCrop(size=(image_size, image_size), scale=(0.6, 1.0), ratio=(0.85, 1.15), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.6),
        A.HueSaturationValue(hue_shift_limit=12, sat_shift_limit=20, val_shift_limit=12, p=0.4),
        A.ImageCompression(quality_range=(40, 90), p=0.5),   # simula ri-salvataggi JPEG
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class ContrastiveObjectDataset(Dataset):
    def __init__(self, root_dir, split: str, image_size: int = IMAGE_SIZE, transforms=None):
        self.split_dir = Path(root_dir) / split
        self.image_files = sorted(
            [p for p in self.split_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
        )
        self.transforms = transforms if transforms is not None else get_contrastive_transforms(image_size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # due augmentation INDIPENDENTI dello stesso oggetto (view A, view B)
        view_a = self.transforms(image=image)["image"]
        view_b = self.transforms(image=image)["image"]

        return {
            "view_a": view_a,
            "view_b": view_b,
            "filename": img_path.name,
        }


def _sanity_check():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = ContrastiveObjectDataset(EXTRACTED_OBJECTS_DIR, SPLIT_TO_TEST, image_size=IMAGE_SIZE)
    print(f"Split '{SPLIT_TO_TEST}': {len(dataset)} oggetti estratti")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    print("view_a shape:", batch["view_a"].shape)
    print("view_b shape:", batch["view_b"].shape)

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    n = len(batch["view_a"])
    fig, axes = plt.subplots(n, 2, figsize=(5, 2.5 * n))
    for i in range(n):
        for col, key in enumerate(["view_a", "view_b"]):
            img = batch[key][i] * std + mean
            img = img.clamp(0, 1).permute(1, 2, 0).numpy()
            axes[i, col].imshow(img)
            axes[i, col].set_title(f"{batch['filename'][i]} - {key}", fontsize=7)
            axes[i, col].axis("off")

    plt.tight_layout()
    out_path = Path("reports/eda") / f"contrastive_sanity_check_{SPLIT_TO_TEST}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"Visualizzazione salvata in: {out_path.resolve()}")


if __name__ == "__main__":
    _sanity_check()