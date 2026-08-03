"""
evaluate.py

Valutazione finale del modello di segmentazione sul TEST set.

Da usare una sola volta, a modello di segmentazione gia' scelto in base
al validation set (non usare i risultati qui per tornare a modificare
iperparametri del modello: il test set deve restare una stima onesta
delle prestazioni finali).

Genera in --output_dir (reports/eval):
    - test_metrics.txt              metriche aggregate (loss, dice, iou)
    - test_predictions_grid.png     griglia: immagine | GT mask | pred mask | overlay confronto
                                     con bounding box della regione predetta disegnato sull'immagine

Uso:
    python evaluate.py
    (modifica i parametri in cima al file se serve)
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import ManipulationDataset, IMAGENET_MEAN, IMAGENET_STD
from model import build_model
from train import BCEDiceLoss, dice_coefficient, iou_score

# ============================================================
# PARAMETRI DI CONFIGURAZIONE (modifica qui se serve)
# ============================================================
DATA_DIR = "data"
IMAGE_SIZE = 512
CHECKPOINT_PATH = "checkpoints/best_model_finetuned.pth"

BATCH_SIZE = 8
NUM_WORKERS = 4
BCE_WEIGHT = 0.5
PRED_THRESHOLD = 0.5     # soglia sulla probabilita' (sigmoid) per binarizzare la predizione

OUTPUT_DIR = "reports/eval"
N_VISUAL_SAMPLES = 8
SEED = 42


def get_bounding_box(binary_mask: np.ndarray):
    """Bounding box (x_min, y_min, x_max, y_max) attorno a tutti i pixel bianchi.
    Ritorna None se la maschera e' completamente vuota."""
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def evaluate_metrics(model, loader, criterion, device):
    """Calcola loss/dice/iou medi su tutto il test set."""
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    n_batches = len(loader)

    with torch.no_grad():
        for batch in tqdm(loader, desc="Valutazione test set"):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            logits = model(images)
            loss = criterion(logits, masks)
            probs = torch.sigmoid(logits)

            total_loss += loss.item()
            total_dice += dice_coefficient(probs, masks).item()
            total_iou += iou_score(probs, masks, threshold=PRED_THRESHOLD).item()

    return {
        "loss": total_loss / n_batches,
        "dice": total_dice / n_batches,
        "iou": total_iou / n_batches,
    }


def save_prediction_grid(model, dataset, device, output_path: Path, n_samples: int, seed: int):
    """Griglia: immagine | GT mask | pred mask | overlay confronto (con bbox disegnato)."""
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    mean = np.array(IMAGENET_MEAN).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD).reshape(1, 1, 3)

    fig, axes = plt.subplots(len(indices), 4, figsize=(14, 3.2 * len(indices)))
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    model.eval()
    with torch.no_grad():
        for row, idx in enumerate(indices):
            sample = dataset[idx]
            image_t = sample["image"].unsqueeze(0).to(device)
            gt_mask = sample["mask"][0].numpy()  # (H, W), 0/1

            logits = model(image_t)
            prob_mask = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred_mask = (prob_mask > PRED_THRESHOLD).astype(np.uint8)

            # de-normalizza l'immagine per la visualizzazione
            img = sample["image"].permute(1, 2, 0).numpy()
            img = (img * std + mean).clip(0, 1)

            # overlay di confronto: verde = solo GT, rosso = solo pred, giallo = sovrapposizione
            overlay = img.copy()
            only_gt = (gt_mask == 1) & (pred_mask == 0)
            only_pred = (gt_mask == 0) & (pred_mask == 1)
            both = (gt_mask == 1) & (pred_mask == 1)
            overlay[only_gt] = [0, 1, 0]
            overlay[only_pred] = [1, 0, 0]
            overlay[both] = [1, 1, 0]

            # immagine con bounding box della predizione (anteprima estrazione oggetto)
            img_with_bbox = (img * 255).astype(np.uint8).copy()
            bbox = get_bounding_box(pred_mask)
            if bbox is not None:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(img_with_bbox, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)

            sample_dice = dice_coefficient(
                torch.tensor(prob_mask).unsqueeze(0).unsqueeze(0),
                torch.tensor(gt_mask).unsqueeze(0).unsqueeze(0),
            ).item()

            axes[row, 0].imshow(img)
            axes[row, 0].set_title(sample["filename"], fontsize=8)
            axes[row, 1].imshow(gt_mask, cmap="gray")
            axes[row, 1].set_title("GT mask", fontsize=8)
            axes[row, 2].imshow(pred_mask, cmap="gray")
            axes[row, 2].set_title(f"Pred mask (dice={sample_dice:.3f})", fontsize=8)
            axes[row, 3].imshow(img_with_bbox)
            axes[row, 3].set_title("Overlay + bbox estrazione", fontsize=8)

            for ax in axes[row]:
                ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"Griglia predizioni salvata in: {output_path.resolve()}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = Path(CHECKPOINT_PATH)
    assert checkpoint_path.exists(), f"Checkpoint non trovato: {checkpoint_path.resolve()}"

    model = build_model(freeze_encoder=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint caricato: {checkpoint_path} (val_dice originale: {checkpoint['val_dice']:.4f})\n")

    test_dataset = ManipulationDataset(DATA_DIR, split="test", image_size=IMAGE_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Test set: {len(test_dataset)} campioni\n")

    criterion = BCEDiceLoss(bce_weight=BCE_WEIGHT)
    metrics = evaluate_metrics(model, test_loader, criterion, device)

    summary = (
        f"===== METRICHE FINALI SU TEST SET =====\n"
        f"Loss: {metrics['loss']:.4f}\n"
        f"Dice: {metrics['dice']:.4f}\n"
        f"IoU:  {metrics['iou']:.4f}\n"
    )
    print("\n" + summary)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "test_metrics.txt", "w") as f:
        f.write(summary)

    save_prediction_grid(model, test_dataset, device, output_dir / "test_predictions_grid.png",
                          n_samples=N_VISUAL_SAMPLES, seed=SEED)


if __name__ == "__main__":
    main()