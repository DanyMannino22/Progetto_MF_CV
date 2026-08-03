"""
train.py

Training loop per il modello di manipulation localization.

Gestisce:
    - loss combinata BCE + Dice (gestisce bene lo sbilanciamento per-immagine)
    - metriche di monitoraggio: Dice coefficient, IoU (loggate su TensorBoard per-epoca
      e stampate a console; usate anche per scegliere il best model)
    - training in mixed precision (AMP) se GPU disponibile
    - salvataggio checkpoint (best + last) in checkpoints/
    - logging su TensorBoard in runs/:
        - per EPOCA: loss, dice, iou (train e val) -> tag "loss/train", "dice/val", ecc.
        - per STEP/batch: solo loss (ogni LOG_EVERY_N_STEPS batch) -> tag "loss_step/train", "loss_step/val"

Uso:
    python train.py
    (modifica i parametri in cima al file se serve)

Per visualizzare i log durante/dopo il training:
    tensorboard --logdir runs
"""

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import ManipulationDataset
from model import build_model

# ============================================================
# PARAMETRI DI CONFIGURAZIONE (modifica qui se serve)
# ============================================================
DATA_DIR = "data"
IMAGE_SIZE = 512

BATCH_SIZE = 8
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 50
BCE_WEIGHT = 0.5          # peso della componente BCE nella loss combinata (1 - questo = peso Dice)
SEED = 42

USE_AMP = True            # mixed precision, solo se CUDA disponibile
EARLY_STOPPING_PATIENCE = 10   # numero di epoche senza miglioramento prima di fermarsi (None per disattivare)

LOG_EVERY_N_STEPS = 10    # ogni quanti batch loggare la loss "per step" su TensorBoard (1 = ogni batch)

CHECKPOINTS_DIR = "checkpoints"
RUNS_DIR = "runs"


# ============================================================
# LOSS E METRICHE
# ============================================================
def dice_coefficient(probs: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """Dice coefficient medio sul batch. probs e targets: (B, 1, H, W), valori in [0,1]."""
    probs = probs.reshape(probs.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    intersection = (probs * targets).sum(dim=1)
    dice = (2 * intersection + smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + smooth)
    return dice.mean()


def iou_score(probs: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, smooth: float = 1e-6) -> torch.Tensor:
    """IoU medio sul batch, calcolato su predizioni binarizzate a `threshold`."""
    preds = (probs > threshold).float()
    preds = preds.reshape(preds.size(0), -1)
    targets = targets.reshape(targets.size(0), -1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


class BCEDiceLoss(nn.Module):
    """Combinazione pesata di BCEWithLogitsLoss (numericamente stabile) e Dice Loss."""

    def __init__(self, bce_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dice_loss = 1.0 - dice_coefficient(probs, targets)
        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


# ============================================================
# UTILITY
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, scaler, is_train: bool,
              writer=None, step_counter=None, log_every: int = 10):
    """
    step_counter: dict mutabile tipo {"train": int, "val": int}, usato per tenere
    un contatore di step GLOBALE (che continua a crescere tra epoche diverse),
    cosi' i grafici 'per step' su TensorBoard mostrano l'andamento continuo
    lungo tutto il training, non solo dentro la singola epoca.
    """
    model.train() if is_train else model.eval()
    phase_key = "train" if is_train else "val"

    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    n_batches = len(loader)

    phase_name = "Train" if is_train else "Val  "
    progress = tqdm(loader, desc=phase_name, leave=False)

    for batch in progress:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=(USE_AMP and device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, masks)

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        with torch.no_grad():
            probs = torch.sigmoid(logits.float())
            dice = dice_coefficient(probs, masks)
            iou = iou_score(probs, masks)

        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()
        progress.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")

        # --- Logging per-step (batch) su TensorBoard ---
        if writer is not None and step_counter is not None:
            step_counter[phase_key] += 1
            if step_counter[phase_key] % log_every == 0:
                writer.add_scalar(f"loss_step/{phase_key}", loss.item(), step_counter[phase_key])

    return {
        "loss": total_loss / n_batches,
        "dice": total_dice / n_batches,
        "iou": total_iou / n_batches,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset & DataLoader ---
    train_dataset = ManipulationDataset(DATA_DIR, split="train", image_size=IMAGE_SIZE)
    val_dataset = ManipulationDataset(DATA_DIR, split="val", image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_dataset)} campioni, {len(train_loader)} batch")
    print(f"Val:   {len(val_dataset)} campioni, {len(val_loader)} batch")

    # --- Modello ---
    model = build_model().to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Parametri allenabili: {sum(p.numel() for p in trainable_params):,}")

    # --- Loss, optimizer, scheduler ---
    criterion = BCEDiceLoss(bce_weight=BCE_WEIGHT)
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    # --- Logging e checkpoint ---
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RUNS_DIR) / run_name
    checkpoints_dir = Path(CHECKPOINTS_DIR)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    print(f"TensorBoard log: {run_dir}")
    print(f"Checkpoint dir:  {checkpoints_dir.resolve()}\n")

    best_val_dice = -1.0
    epochs_without_improvement = 0
    step_counter = {"train": 0, "val": 0}

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"--- Epoch {epoch}/{NUM_EPOCHS} ---")

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler, is_train=True,
                                   writer=writer, step_counter=step_counter, log_every=LOG_EVERY_N_STEPS)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, scaler, is_train=False,
                                 writer=writer, step_counter=step_counter, log_every=LOG_EVERY_N_STEPS)

        scheduler.step(val_metrics["dice"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train -> loss: {train_metrics['loss']:.4f}  dice: {train_metrics['dice']:.4f}  iou: {train_metrics['iou']:.4f}")
        print(f"Val   -> loss: {val_metrics['loss']:.4f}  dice: {val_metrics['dice']:.4f}  iou: {val_metrics['iou']:.4f}")
        print(f"LR: {current_lr:.2e}")

        # --- Logging TensorBoard (per-epoca) ---
        writer.add_scalar("loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("dice/train", train_metrics["dice"], epoch)
        writer.add_scalar("dice/val", val_metrics["dice"], epoch)
        writer.add_scalar("iou/train", train_metrics["iou"], epoch)
        writer.add_scalar("iou/val", val_metrics["iou"], epoch)
        writer.add_scalar("lr", current_lr, epoch)

        # --- Checkpoint ---
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_metrics["dice"],
            "val_loss": val_metrics["loss"],
        }
        torch.save(checkpoint, checkpoints_dir / "last_model.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            torch.save(checkpoint, checkpoints_dir / "best_model.pth")
            print(f"Nuovo best model salvato (val_dice: {best_val_dice:.4f})")
        else:
            epochs_without_improvement += 1

        print()

        # --- Early stopping ---
        if EARLY_STOPPING_PATIENCE is not None and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping: nessun miglioramento da {EARLY_STOPPING_PATIENCE} epoche.")
            break

    writer.close()
    print(f"\nTraining completato. Miglior val_dice: {best_val_dice:.4f}")
    print(f"Modello migliore salvato in: {(checkpoints_dir / 'best_model.pth').resolve()}")


if __name__ == "__main__":
    main()