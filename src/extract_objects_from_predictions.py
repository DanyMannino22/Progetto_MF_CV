"""
extract_objects_from_predictions.py

Versione "reale" della pipeline: usa il modello di segmentazione ALLENATO
per predire la maschera (invece di leggerla da file GT), poi estrae il
bounding box e il crop dell'oggetto - esattamente il flusso che si userebbe
in produzione su immagini mai viste, dove la ground truth non esiste.

Passaggi per ogni immagine:
    1. resize a IMAGE_SIZE per il modello (es. 512x512)
    2. forward pass -> mappa di probabilita' 512x512
    3. resize della mappa di probabilita' alla risoluzione ORIGINALE dell'immagine
       (fondamentale: il bbox deve essere calcolato in coordinate reali, non
       nella griglia 512x512 usata solo internamente dal modello)
    4. binarizzazione + bounding box + padding
    5. crop dall'immagine originale a piena risoluzione

Uso:
    python extract_objects_from_predictions.py
    (modifica i parametri in cima al file se serve)
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dataset import get_eval_transforms
from model import build_model
from extract_objects import get_bounding_box, apply_padding, IMG_EXTENSIONS

# ============================================================
# PARAMETRI DI CONFIGURAZIONE (modifica qui se serve)
# ============================================================
DATA_DIR = "data"
SPLITS = ["train", "val", "test"]     # su quali split girare l'inferenza

CHECKPOINT_PATH = "checkpoints/best_model_finetuned.pth"
IMAGE_SIZE = 512

OUTPUT_DIR = "data/extracted_objects_predicted"
PRED_THRESHOLD = 0.5        # soglia sulla probabilita' per binarizzare la maschera predetta
PADDING_RATIO = 0.10        # stesso padding usato nell'estrazione da GT, per coerenza
MIN_BBOX_AREA_PX = 100

SAVE_PREDICTED_MASK = True  # se True, salva anche la maschera predetta (risoluzione originale) per debug


def predict_mask(model, image_bgr: np.ndarray, transforms, device) -> np.ndarray:
    """Ritorna la mappa di probabilita' (float, 0-1) alla risoluzione ORIGINALE dell'immagine."""
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    augmented = transforms(image=image_rgb)
    image_t = augmented["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_t)
        prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()   # (IMAGE_SIZE, IMAGE_SIZE)

    # resize della probabilita' alla risoluzione originale (interpolazione lineare,
    # poi si binarizza DOPO, cosi' il bordo si adatta meglio alla risoluzione reale)
    prob_map_orig = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return prob_map_orig


def process_split(split: str, model, transforms, device, data_dir: Path, output_dir: Path):
    images_dir = data_dir / split / "images"
    split_output_dir = output_dir / split
    split_output_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_PREDICTED_MASK:
        masks_output_dir = split_output_dir / "predicted_masks"
        masks_output_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])

    records = []
    skipped_empty = 0
    skipped_too_small = 0

    for img_path in tqdm(image_files, desc=f"Inferenza+estrazione {split}"):
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        prob_map = predict_mask(model, image, transforms, device)
        binary_mask = (prob_map > PRED_THRESHOLD).astype(np.uint8)

        if SAVE_PREDICTED_MASK:
            cv2.imwrite(str(masks_output_dir / img_path.name), binary_mask * 255)

        bbox = get_bounding_box(binary_mask)
        if bbox is None:
            skipped_empty += 1
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

    csv_path = split_output_dir / "bboxes_predicted.csv"
    if records:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)

    print(f"[{split}] Estratti: {len(records)} | Maschere vuote predette: {skipped_empty} | "
          f"Troppo piccoli scartati: {skipped_too_small}")

    return records


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = Path(CHECKPOINT_PATH)
    assert checkpoint_path.exists(), f"Checkpoint non trovato: {checkpoint_path.resolve()}"

    model = build_model(freeze_encoder=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Checkpoint caricato: {checkpoint_path} (val_dice: {checkpoint['val_dice']:.4f})\n")

    transforms = get_eval_transforms(IMAGE_SIZE)   # solo resize + normalizzazione, no augmentation

    data_dir = Path(DATA_DIR)
    output_dir = Path(OUTPUT_DIR)

    for split in SPLITS:
        process_split(split, model, transforms, device, data_dir, output_dir)

    print(f"\nOggetti estratti (da predizioni) salvati in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()