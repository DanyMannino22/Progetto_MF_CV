"""
train_metric.py

Training del modello di metric learning tramite instance discrimination
(NT-Xent contrastive loss, come in SimCLR).

Metrica di monitoraggio: "matching accuracy" - per ogni view A di un
oggetto, controlla se la sua view B (la vera coppia positiva) e' quella
con similarita' piu' alta tra TUTTI gli altri elementi del batch (view A
e B di tutti gli altri oggetti). E' una metrica molto piu' interpretabile
della loss grezza: dice letteralmente "quante volte il modello indovina
l'oggetto giusto tra tutti gli altri nel batch".

Uso:
    python train_metric.py
    (modifica i parametri in cima al file se serve)

Per visualizzare i log:
    tensorboard --logdir runs
"""

from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_metric import ContrastiveObjectDataset, IMAGE_SIZE
from model_metric import ContrastiveModel
from train import set_seed   # riuso la funzione di seeding gia' scritta per la segmentazione

# ============================================================
# PARAMETRI DI CONFIGURAZIONE (modifica qui se serve)
# ============================================================
EXTRACTED_OBJECTS_DIR = "data/extracted_objects"

BATCH_SIZE = 128          # batch grande = piu' negativi "gratis" per la NT-Xent, meglio se possibile
NUM_WORKERS = 4
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
TEMPERATURE = 0.2         # temperatura NT-Xent: valori tipici 0.1-0.5, piu' basso = distribuzione piu' "dura"
SEED = 42

EARLY_STOPPING_PATIENCE = 15
SCHEDULER_PATIENCE = 7

USE_AMP = True     # riduce la memoria GPU, permette batch size piu' alti a parita' di hardware

LOG_EVERY_N_STEPS = 10

CHECKPOINTS_DIR = "checkpoints"
RUNS_DIR = "runs"
BEST_MODEL_NAME = "best_metric_model.pth"
LAST_MODEL_NAME = "last_metric_model.pth"


def nt_xent_loss(z_a: torch.Tensor, z_b: torch.Tensor, temperature: float):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy), come in SimCLR.

    z_a, z_b: (B, D) embedding normalizzati L2, view A e B dello stesso batch di oggetti.
    Ritorna: loss scalare, accuracy di matching (top-1 tra tutti i negativi nel batch)
    """
    batch_size = z_a.size(0)
    device = z_a.device

    # concatena le due view: (2B, D). Le prime B righe sono le view A, le seconde B le view B
    z = torch.cat([z_a, z_b], dim=0)

    # matrice di similarita' coseno (gia' normalizzati -> il prodotto scalare E' la cosine similarity)
    sim_matrix = torch.matmul(z, z.T) / temperature   # (2B, 2B)

    # maschera la diagonale (similarita' di un elemento con se stesso) con -inf, non e' un negativo valido
    self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
    sim_matrix.masked_fill_(self_mask, float("-inf"))

    # per la riga i in [0, B-1] (view A), il positivo e' alla posizione i+B (la sua view B)
    # per la riga i in [B, 2B-1] (view B), il positivo e' alla posizione i-B (la sua view A)
    positive_indices = torch.arange(2 * batch_size, device=device)
    positive_indices = (positive_indices + batch_size) % (2 * batch_size)

    loss = F.cross_entropy(sim_matrix, positive_indices)

    with torch.no_grad():
        predictions = sim_matrix.argmax(dim=1)
        accuracy = (predictions == positive_indices).float().mean()

    return loss, accuracy


def run_epoch(model, loader, optimizer, device, temperature, is_train: bool, scaler=None,
              writer=None, step_counter=None, log_every: int = 10):
    model.train() if is_train else model.eval()
    phase_key = "train" if is_train else "val"

    total_loss, total_acc = 0.0, 0.0
    n_batches = len(loader)

    phase_name = "Train" if is_train else "Val  "
    progress = tqdm(loader, desc=phase_name, leave=False)

    for batch in progress:
        view_a = batch["view_a"].to(device, non_blocking=True)
        view_b = batch["view_b"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device.type, enabled=(USE_AMP and device.type == "cuda")):
                z_a = model(view_a)
                z_b = model(view_b)
                loss, accuracy = nt_xent_loss(z_a, z_b, temperature)

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy.item()
        progress.set_postfix(loss=f"{loss.item():.4f}", acc=f"{accuracy.item():.3f}")

        if writer is not None and step_counter is not None:
            step_counter[phase_key] += 1
            if step_counter[phase_key] % log_every == 0:
                writer.add_scalar(f"loss_step/{phase_key}", loss.item(), step_counter[phase_key])

    return {"loss": total_loss / n_batches, "accuracy": total_acc / n_batches}


def main():
    assert EARLY_STOPPING_PATIENCE > SCHEDULER_PATIENCE, \
        "EARLY_STOPPING_PATIENCE deve essere maggiore di SCHEDULER_PATIENCE."

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_dataset = ContrastiveObjectDataset(EXTRACTED_OBJECTS_DIR, split="train", image_size=IMAGE_SIZE)
    val_dataset = ContrastiveObjectDataset(EXTRACTED_OBJECTS_DIR, split="val", image_size=IMAGE_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)

    print(f"Train: {len(train_dataset)} oggetti, {len(train_loader)} batch")
    print(f"Val:   {len(val_dataset)} oggetti, {len(val_loader)} batch")

    model = ContrastiveModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=SCHEDULER_PATIENCE)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    run_name = "metric_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(RUNS_DIR) / run_name
    checkpoints_dir = Path(CHECKPOINTS_DIR)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))

    print(f"TensorBoard log: {run_dir}")
    print(f"Checkpoint dir:  {checkpoints_dir.resolve()}\n")

    best_val_accuracy = -1.0
    epochs_without_improvement = 0
    step_counter = {"train": 0, "val": 0}

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"--- Epoch {epoch}/{NUM_EPOCHS} ---")

        train_metrics = run_epoch(model, train_loader, optimizer, device, TEMPERATURE, is_train=True, scaler=scaler,
                                   writer=writer, step_counter=step_counter, log_every=LOG_EVERY_N_STEPS)
        val_metrics = run_epoch(model, val_loader, optimizer, device, TEMPERATURE, is_train=False, scaler=scaler,
                                 writer=writer, step_counter=step_counter, log_every=LOG_EVERY_N_STEPS)

        scheduler.step(val_metrics["accuracy"])
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Train -> loss: {train_metrics['loss']:.4f}  matching_acc: {train_metrics['accuracy']:.4f}")
        print(f"Val   -> loss: {val_metrics['loss']:.4f}  matching_acc: {val_metrics['accuracy']:.4f}")
        print(f"LR: {current_lr:.2e}")

        writer.add_scalar("loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("matching_accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("matching_accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("lr", current_lr, epoch)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_accuracy": val_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
        }
        torch.save(checkpoint, checkpoints_dir / LAST_MODEL_NAME)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            epochs_without_improvement = 0
            torch.save(checkpoint, checkpoints_dir / BEST_MODEL_NAME)
            print(f"Nuovo best model salvato (val_matching_acc: {best_val_accuracy:.4f})")
        else:
            epochs_without_improvement += 1

        print()

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping: nessun miglioramento da {EARLY_STOPPING_PATIENCE} epoche.")
            break

    writer.close()
    print(f"\nTraining completato. Miglior val_matching_accuracy: {best_val_accuracy:.4f}")
    print(f"Modello migliore salvato in: {(checkpoints_dir / BEST_MODEL_NAME).resolve()}")


if __name__ == "__main__":
    main()