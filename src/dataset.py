"""
dataset.py

Dataset PyTorch per il task di manipulation localization.

Gestisce:
    - lettura immagine (RGB) e maschera (grayscale) allineate per nome file
    - resize a dimensione fissa (bicubic per immagine, nearest per maschera
      cosi' da non introdurre nuovi valori intermedi nella maschera)
    - binarizzazione della maschera post-resize (soglia 127)
    - augmentation (solo su split 'train') sincronizzata image+mask via albumentations
    - normalizzazione ImageNet (necessaria per l'encoder ResNet pretrained)

Uso rapido (test standalone):
    python dataset.py
    (modifica DATA_DIR, IMAGE_SIZE, SPLIT_TO_TEST ecc. qui sotto se serve)
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============================================================
# PARAMETRI DI CONFIGURAZIONE (modifica qui se serve)
# ============================================================
DATA_DIR = "data"          # cartella contenente train/ val/ test/
IMAGE_SIZE = 512            # lato del resize quadrato
MASK_THRESHOLD = 127        # soglia binarizzazione maschera (0-255)
SPLIT_TO_TEST = "train"     # split usato dal sanity check quando lanci questo file direttamente
SANITY_CHECK_OUTPUT_DIR = "reports/eda"

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int):
    """Augmentation per il training: geometriche + fotometriche leggere,
    applicate in modo sincronizzato a immagine e maschera da albumentations."""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC,
                  mask_interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
                            border_mode=cv2.BORDER_CONSTANT, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_eval_transforms(image_size: int):
    """Nessuna augmentation per val/test: solo resize deterministico + normalizzazione."""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC,
                  mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class ManipulationDataset(Dataset):
    def __init__(self, root_dir, split: str, image_size: int = 512, transforms=None):
        """
        Args:
            root_dir: path alla cartella 'data' (contenente train/val/test)
            split: 'train', 'val' o 'test'
            image_size: lato del resize quadrato
            transforms: se None, usa le trasformazioni di default in base allo split
                        (augmentation per 'train', solo resize+norm per val/test)
        """
        self.split_dir = Path(root_dir) / split
        self.images_dir = self.split_dir / "images"
        self.masks_dir = self.split_dir / "masks"

        self.image_files = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
        )
        # mappa nome-senza-estensione -> path maschera, per gestire estensioni diverse
        mask_lookup = {
            p.stem: p for p in self.masks_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS
        }
        self.mask_files = [mask_lookup[p.stem] for p in self.image_files]

        if transforms is not None:
            self.transforms = transforms
        elif split == "train":
            self.transforms = get_train_transforms(image_size)
        else:
            self.transforms = get_eval_transforms(image_size)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        augmented = self.transforms(image=image, mask=mask)
        image_t = augmented["image"]              # tensor (3, H, W), float, normalizzato
        mask_t = augmented["mask"]                 # tensor (H, W), uint8/float, 0-255 dopo resize

        # binarizzazione post-resize (nearest neighbor non introduce nuovi valori,
        # ma la maschera originale poteva gia' avere valori intermedi da antialiasing)
        mask_t = (mask_t > MASK_THRESHOLD).float().unsqueeze(0)   # -> (1, H, W), valori {0., 1.}

        return {
            "image": image_t,
            "mask": mask_t,
            "filename": img_path.name,
        }


def _sanity_check(data_dir: str, split: str, image_size: int, output_dir: str):
    """Piccolo test: carica alcuni campioni, stampa shape/range, salva una visualizzazione."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dataset = ManipulationDataset(data_dir, split, image_size=image_size)
    print(f"Split '{split}': {len(dataset)} campioni")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    print("image batch shape:", batch["image"].shape, batch["image"].dtype)
    print("mask batch shape:", batch["mask"].shape, batch["mask"].dtype)
    print("image value range:", batch["image"].min().item(), batch["image"].max().item())
    print("mask unique values:", torch.unique(batch["mask"]).tolist())

    # de-normalizza per visualizzare correttamente
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    fig, axes = plt.subplots(len(batch["image"]), 2, figsize=(6, 3 * len(batch["image"])))
    for i in range(len(batch["image"])):
        img = batch["image"][i] * std + mean
        img = img.clamp(0, 1).permute(1, 2, 0).numpy()
        mask = batch["mask"][i, 0].numpy()

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(batch["filename"][i], fontsize=8)
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("mask (binaria)", fontsize=8)
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    out_path = Path(output_dir) / f"dataset_sanity_check_{split}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=120)
    print(f"Visualizzazione salvata in: {out_path.resolve()}")


if __name__ == "__main__":
    # Nessun argomento da terminale: modifica i parametri in cima al file se serve.
    _sanity_check(
        data_dir=DATA_DIR,
        split=SPLIT_TO_TEST,
        image_size=IMAGE_SIZE,
        output_dir=SANITY_CHECK_OUTPUT_DIR,
    )