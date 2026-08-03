"""
inference.py

Pipeline di inferenza END-TO-END su una NUOVA immagine (mai vista nel
training), che mette insieme tutti e tre gli step del progetto:

    1. Modello di segmentazione -> predice la maschera dell'area manipolata
    2. Bounding box + crop dell'oggetto dall'immagine originale
    3. Modello di metric learning -> embedding dell'oggetto estratto
    4. Confronto con la GALLERIA di embedding gia' calcolata da
       cluster_objects.py -> trova gli oggetti piu' simili nel dataset
       e decide se e' un match (stessa istanza gia' vista altrove) o
       un oggetto unico (mai visto)

Genera in --output_dir (reports/inference):
    - <nome_immagine>_result.png   visualizzazione completa del risultato

Uso:
    python inference.py --image path/alla/nuova_immagine.jpg
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import get_eval_transforms, IMAGENET_MEAN, IMAGENET_STD
from model import build_model as build_segmentation_model
from model_metric import ContrastiveModel
from extract_objects import get_bounding_box, apply_padding
from cluster_objects import get_deterministic_transform

# ============================================================
# PARAMETRI DI CONFIGURAZIONE (modifica qui se serve)
# ============================================================
SEGMENTATION_CHECKPOINT = "checkpoints/best_model_finetuned.pth"
METRIC_CHECKPOINT = "checkpoints/best_metric_model.pth"

SEGMENTATION_IMAGE_SIZE = 512
METRIC_IMAGE_SIZE = 224

PRED_MASK_THRESHOLD = 0.5     # soglia sigmoid per binarizzare la maschera predetta
PADDING_RATIO = 0.10           # stesso padding usato in extract_objects.py

GALLERY_DIR = "reports/clustering"    # dove cluster_objects.py ha salvato la galleria
MATCH_SIMILARITY_THRESHOLD = 0.85     # sopra questa soglia = "match trovato" (tarata sull'istogramma)
TOP_K_MATCHES = 5                      # quanti oggetti simili mostrare nella visualizzazione

OUTPUT_DIR = "reports/inference"


def load_gallery(gallery_dir: Path):
    embeddings = np.load(gallery_dir / "gallery_embeddings.npy")
    metadata = []
    with open(gallery_dir / "gallery_metadata.csv", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metadata.append(row)
    return embeddings, metadata


def predict_mask_and_bbox(image_bgr, seg_model, device):
    """Segmentazione + bbox, stessa logica di extract_objects_from_predictions.py"""
    orig_h, orig_w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    transforms = get_eval_transforms(SEGMENTATION_IMAGE_SIZE)
    image_t = transforms(image=image_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = seg_model(image_t)
        prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()

    prob_map_orig = cv2.resize(prob_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    binary_mask = (prob_map_orig > PRED_MASK_THRESHOLD).astype(np.uint8)

    bbox = get_bounding_box(binary_mask)
    return binary_mask, bbox


def compute_embedding(crop_bgr, metric_model, device):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    transform = get_deterministic_transform(METRIC_IMAGE_SIZE)
    crop_t = transform(image=crop_rgb)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = metric_model(crop_t)[0].cpu().numpy()
    return embedding


def find_matches(query_embedding, gallery_embeddings, gallery_metadata, top_k):
    similarities = gallery_embeddings @ query_embedding   # embedding gia' L2-normalizzati -> cosine sim
    top_indices = np.argsort(-similarities)[:top_k]
    matches = [(gallery_metadata[i], float(similarities[i])) for i in top_indices]
    return matches


def visualize_result(image_bgr, binary_mask, bbox, crop_bgr, matches, output_path: Path, image_name: str):
    n_matches = len(matches)
    fig, axes = plt.subplots(1, 3 + n_matches, figsize=(3.2 * (3 + n_matches), 3.5))

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_with_bbox = image_rgb.copy()
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        cv2.rectangle(image_with_bbox, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

    axes[0].imshow(image_rgb)
    axes[0].set_title("Immagine originale", fontsize=8)
    axes[1].imshow(binary_mask, cmap="gray")
    axes[1].set_title("Maschera predetta", fontsize=8)
    axes[2].imshow(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Oggetto estratto", fontsize=8)

    for ax in axes[:3]:
        ax.axis("off")

    for i, (meta, sim) in enumerate(matches):
        match_img = cv2.cvtColor(cv2.imread(meta["path"]), cv2.COLOR_BGR2RGB)
        ax = axes[3 + i]
        ax.imshow(match_img)
        is_match = sim >= MATCH_SIMILARITY_THRESHOLD
        color = "green" if is_match else "gray"
        ax.set_title(f"sim={sim:.3f}\ncluster={meta['cluster_id']}", fontsize=8, color=color)
        ax.axis("off")

    plt.suptitle(f"Risultato inferenza: {image_name}")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Inferenza end-to-end su una nuova immagine")
    parser.add_argument("--image", type=str, required=True, help="Path all'immagine da analizzare")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    image_path = Path(args.image)
    assert image_path.exists(), f"Immagine non trovata: {image_path}"

    image_bgr = cv2.imread(str(image_path))
    assert image_bgr is not None, f"Impossibile leggere l'immagine: {image_path}"

    # --- Carica i modelli ---
    seg_model = build_segmentation_model(freeze_encoder=False).to(device)
    seg_checkpoint = torch.load(SEGMENTATION_CHECKPOINT, map_location=device)
    seg_model.load_state_dict(seg_checkpoint["model_state_dict"])
    seg_model.eval()

    metric_model = ContrastiveModel().to(device)
    metric_checkpoint = torch.load(METRIC_CHECKPOINT, map_location=device)
    metric_model.load_state_dict(metric_checkpoint["model_state_dict"])
    metric_model.eval()

    print("Modelli caricati.\n")

    # --- Step 1-2: segmentazione + bbox ---
    binary_mask, bbox = predict_mask_and_bbox(image_bgr, seg_model, device)

    if bbox is None:
        print("Nessuna area manipolata rilevata in questa immagine (maschera predetta vuota).")
        return

    x_min, y_min, x_max, y_max = apply_padding(bbox, image_bgr.shape, PADDING_RATIO)
    crop_bgr = image_bgr[y_min:y_max + 1, x_min:x_max + 1]
    print(f"Area manipolata rilevata: bbox={bbox}")

    # --- Step 3: embedding ---
    query_embedding = compute_embedding(crop_bgr, metric_model, device)

    # --- Step 4: confronto con la galleria ---
    gallery_dir = Path(GALLERY_DIR)
    assert (gallery_dir / "gallery_embeddings.npy").exists(), \
        "Galleria non trovata: esegui prima cluster_objects.py per generarla."

    gallery_embeddings, gallery_metadata = load_gallery(gallery_dir)
    matches = find_matches(query_embedding, gallery_embeddings, gallery_metadata, TOP_K_MATCHES)

    best_match, best_similarity = matches[0]

    print(f"\nMiglior corrispondenza: {best_match['filename']} (similarita': {best_similarity:.4f})")
    if best_similarity >= MATCH_SIMILARITY_THRESHOLD:
        print(f"MATCH TROVATO -> l'oggetto sembra corrispondere al cluster {best_match['cluster_id']}")
    else:
        print("NESSUN MATCH -> l'oggetto sembra unico, non risulta simile a nulla nella galleria")

    # --- Visualizzazione ---
    output_dir = Path(OUTPUT_DIR)
    output_path = output_dir / f"{image_path.stem}_result.png"
    visualize_result(image_bgr, binary_mask, bbox, crop_bgr, matches, output_path, image_path.name)
    print(f"\nVisualizzazione salvata in: {output_path.resolve()}")


if __name__ == "__main__":
    main()