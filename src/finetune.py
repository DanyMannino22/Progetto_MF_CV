"""
finetune.py

Fase 2 del training: riparte dal miglior checkpoint ottenuto con encoder
congelato (train.py) e sblocca l'encoder per il fine-tuning completo.

Learning rate differenziato:
    - ENCODER_LR molto basso: l'encoder pretrained va aggiornato con cautela,
      per non distruggere le feature ImageNet gia' utili (catastrophic forgetting)
    - DECODER_LR piu' alto: il decoder puo' continuare ad apprendere piu' aggressivamente

Riusa loss, metriche, run_epoch da train.py per evitare duplicazione di codice.

Uso:
    python finetune.py
    (modifica i parametri in cima al file se serve)

Per visualizzare i log:
    tensorboard --logdir runs
"""

from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ManipulationDataset
from model import build_model
from train import BCEDiceLoss, run_epoch, set_seed

# ============================================================
# PARAMETRI DI CONFIGURAZIONE (modifica qui se serve)
# ============================================================
DATA_DIR = "data"
IMAGE_SIZE = 512

CHECKPOINT_TO_RESUME = "checkpoints/best_model.pth"   # modello con encoder congelato, gia' allenato

BATCH_SIZE = 8
NUM_WORKERS = 4
ENCODER_LR = 1e-5     # LR basso: encoder pretrained, va aggiornato con cautela
DECODER_LR = 1e-4     # LR piu' alto: decoder, puo' continuare ad apprendere piu' velocemente
WEIGHT_DECAY = 1e-5
NUM_EPOCHS = 30
BCE_WEIGHT = 0.5
SEED = 42

USE_AMP = True
SCHEDULER_PATIENCE = 5             # epoche senza miglioramento prima di dimezzare il LR
EARLY_STOPPING_PATIENCE = 15       # DEVE essere > SCHEDULER_PATIENCE, altrimenti il training
                                    # si ferma prima che lo scheduler abbia la possibilita' di agire

LOG_EVERY_N_STEPS = 10

CHECKPOINTS_DIR = "checkpoints"
RUNS_DIR = "runs"
BEST_MODEL_NAME = "best_model_finetuned.pth"
LAST_MODEL_NAME = "last_model_finetuned.pth"


def main():
    assert EARLY_STOPPING_PATIENCE > SCHEDULER_PATIENCE, \
        "EARLY_STOPPING_PATIENCE deve essere maggiore di SCHEDULER_PATIENCE, " \
        "altrimenti l'early stopping scatta prima che lo scheduler possa abbassare il LR."

    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset & DataLoader (identici a train.py) ---
    train_dataset = ManipulationDataset(DATA_DIR, split="train", image_size=IMAGE_SIZE)
    val_dataset = ManipulationDataset(DATA_DIR, split="val", image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Train: {len(train_dataset)} campioni, {len(train_loader)} batch")
    print(f"Val:   {len(val_dataset)} campioni, {len(val_loader)} batch")

    # --- Modello: costruito SENZA congelamento, poi caricati i pesi del checkpoint precedente ---
    model = build_model(freeze_encoder=False).to(device)

    checkpoint_path = Path(CHECKPOINT_TO_RESUME)
    assert checkpoint_path.exists(), f"Checkpoint non trovato: {checkpoint_path.resolve()}"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"\nPesi caricati da: {checkpoint_path}")
    print(f"   (epoca originale: {checkpoint['epoch']}, val_dice: {checkpoint['val_dice']:.4f})")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parametri allenabili ora: {trainable_params:,} / {total_params:,} (100% sbloccati)\n")

    # --- Loss, optimizer con LR differenziato, scheduler ---
    criterion = BCEDiceLoss(bce_weight=BCE_WEIGHT)

    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]

    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": ENCODER_LR},
        {"params": decoder_params, "lr": DECODER_LR},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    # --- Logging e checkpoint ---
    run_name = "finetune_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RUNS_DIR) / run_name
    checkpoints_dir = Path(CHECKPOINTS_DIR)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    print(f"TensorBoard log: {run_dir}")
    print(f"Checkpoint dir:  {checkpoints_dir.resolve()}\n")

    best_val_dice = checkpoint["val_dice"]   # partiamo dal valore gia' raggiunto, non da -1
    print(f"Val_dice di partenza (dal checkpoint): {best_val_dice:.4f}\n")
    epochs_without_improvement = 0
    step_counter = {"train": 0, "val": 0}

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"--- Fine-tune Epoch {epoch}/{NUM_EPOCHS} ---")

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, scaler, is_train=True,
                                   writer=writer, step_counter=step_counter, log_every=LOG_EVERY_N_STEPS)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, scaler, is_train=False,
                                 writer=writer, step_counter=step_counter, log_every=LOG_EVERY_N_STEPS)

        scheduler.step(val_metrics["dice"])
        current_lr_encoder = optimizer.param_groups[0]["lr"]
        current_lr_decoder = optimizer.param_groups[1]["lr"]

        print(f"Train -> loss: {train_metrics['loss']:.4f}  dice: {train_metrics['dice']:.4f}  iou: {train_metrics['iou']:.4f}")
        print(f"Val   -> loss: {val_metrics['loss']:.4f}  dice: {val_metrics['dice']:.4f}  iou: {val_metrics['iou']:.4f}")
        print(f"LR encoder: {current_lr_encoder:.2e}  |  LR decoder: {current_lr_decoder:.2e}")

        # --- Logging TensorBoard (per-epoca) ---
        writer.add_scalar("loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("dice/train", train_metrics["dice"], epoch)
        writer.add_scalar("dice/val", val_metrics["dice"], epoch)
        writer.add_scalar("iou/train", train_metrics["iou"], epoch)
        writer.add_scalar("iou/val", val_metrics["iou"], epoch)
        writer.add_scalar("lr/encoder", current_lr_encoder, epoch)
        writer.add_scalar("lr/decoder", current_lr_decoder, epoch)

        # --- Checkpoint ---
        new_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_dice": val_metrics["dice"],
            "val_loss": val_metrics["loss"],
        }
        torch.save(new_checkpoint, checkpoints_dir / LAST_MODEL_NAME)

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            epochs_without_improvement = 0
            torch.save(new_checkpoint, checkpoints_dir / BEST_MODEL_NAME)
            print(f"Nuovo best model salvato (val_dice: {best_val_dice:.4f})")
        else:
            epochs_without_improvement += 1

        print()

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping: nessun miglioramento da {EARLY_STOPPING_PATIENCE} epoche.")
            break

    writer.close()
    print(f"\nFine-tuning completato. Miglior val_dice: {best_val_dice:.4f}")
    print(f"Modello migliore salvato in: {(checkpoints_dir / BEST_MODEL_NAME).resolve()}")


if __name__ == "__main__":
    main()