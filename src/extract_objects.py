"""
extract_objects.py

Estrae l'oggetto manipolato da ogni immagine, usando la maschera di ground
truth per calcolare un bounding box e ritagliare la regione corrispondente
dall'immagine RGB originale (non dalla maschera).

Per ogni split (train/val/test) salva in data/extracted_objects/<split>/:
    - <filename>.png       il crop RGB dell'oggetto (con padding attorno al bbox)
    - bboxes.csv            metadati: filename, bbox originale, area, dimensioni crop

Il nome del file estratto coincide con quello dell'immagine originale:
    sara' l'identificativo univoco usato nello step di metric learning
    per sapere quali crop appartengono alla stessa istanza (via augmentation)
    e quali sono istanze diverse.

Uso:
    python extract_objects.py
    (modifica i parametri in cima al file se serve)
"""

import csv
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# ============================================================
# PARAMETRI DI CONFIGURAZIONE (modifica qui se serve)
# ============================================================
DATA_DIR = "data"
OUTPUT_DIR = "data/extracted_objects"
SPLITS = ["train", "val", "test"]

MASK_THRESHOLD = 127        # soglia binarizzazione maschera (0-255)
PADDING_RATIO = 0.10        # padding attorno al bbox, come frazione della dimensione del bbox stesso
MIN_BBOX_AREA_PX = 100      # bbox piu' piccoli di questa area (in pixel, sull'immagine originale) vengono scartati

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_bounding_box(binary_mask: np.ndarray):
    """Bounding box (x_min, y_min, x_max, y_max) attorno a tutti i pixel bianchi.
    Ritorna None se la maschera e' completamente vuota."""
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def apply_padding(bbox, image_shape, padding_ratio: float):
    """Allarga il bbox di una percentuale della sua dimensione, clampando ai bordi immagine."""
    x_min, y_min, x_max, y_max = bbox
    h_img, w_img = image_shape[:2]

    box_w = x_max - x_min
    box_h = y_max - y_min
    pad_x = int(box_w * padding_ratio)
    pad_y = int(box_h * padding_ratio)

    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(w_img - 1, x_max + pad_x)
    y_max = min(h_img - 1, y_max + pad_y)

    return x_min, y_min, x_max, y_max


def process_split(split: str, data_dir: Path, output_dir: Path):
    images_dir = data_dir / split / "images"
    masks_dir = data_dir / split / "masks"

    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])
    mask_lookup = {p.stem: p for p in masks_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS}

    records = []
    skipped_empty_mask = 0
    skipped_too_small = 0

    for img_path in tqdm(image_files, desc=f"Estrazione {split}"):
        mask_path = mask_lookup.get(img_path.stem)
        if mask_path is None:
            continue

        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            continue

        binary_mask = (mask > MASK_THRESHOLD).astype(np.uint8)
        bbox = get_bounding_box(binary_mask)

        if bbox is None:
            skipped_empty_mask += 1
            continue

        x_min, y_min, x_max, y_max = bbox
        bbox_area = (x_max - x_min + 1) * (y_max - y_min + 1)
        if bbox_area < MIN_BBOX_AREA_PX:
            skipped_too_small += 1
            continue

        x_min_p, y_min_p, x_max_p, y_max_p = apply_padding(bbox, image.shape, PADDING_RATIO)
        crop = image[y_min_p:y_max_p + 1, x_min_p:x_max_p + 1]

        out_path = split_output_dir / img_path.name
        cv2.imwrite(str(out_path), crop)

        records.append({
            "filename": img_path.name,
            "bbox_x_min": x_min, "bbox_y_min": y_min,
            "bbox_x_max": x_max, "bbox_y_max": y_max,
            "bbox_area_px": bbox_area,
            "crop_width": x_max_p - x_min_p + 1,
            "crop_height": y_max_p - y_min_p + 1,
        })

    # --- CSV di metadati ---
    csv_path = split_output_dir / "bboxes.csv"
    if records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    print(f"[{split}] Estratti: {len(records)} | Maschere vuote saltate: {skipped_empty_mask} | "
          f"Troppo piccoli saltati: {skipped_too_small}")

    return records


def main():
    data_dir = Path(DATA_DIR)
    output_dir = Path(OUTPUT_DIR)

    for split in SPLITS:
        process_split(split, data_dir, output_dir)

    print(f"\nOggetti estratti salvati in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()