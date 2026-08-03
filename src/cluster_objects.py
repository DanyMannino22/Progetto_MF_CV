"""
cluster_objects.py

Confronta TUTTI gli oggetti estratti dal dataset tra loro, usando gli
embedding del modello di metric learning, e li raggruppa in base alla
similarita' (DBSCAN su distanza coseno).

A differenza di visualize_embeddings.py (che verificava l'invarianza
alle augmentation dello STESSO oggetto), qui l'obiettivo e' scoprire
raggruppamenti REALI tra oggetti DIVERSI del dataset che il modello
ritiene sufficientemente simili da essere la "stessa classe"/istanza.

Genera in --output_dir (reports/clustering):
    - clusters.csv                  filename -> cluster_id (-1 = nessun match trovato)
    - cluster_summary.txt            statistiche sui cluster trovati
    - cluster_grid_<id>.png          griglia di immagini per i cluster piu' numerosi

Uso:
    python cluster_objects.py
    (modifica i parametri in cima al file se serve)
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.cluster import AgglomerativeClustering
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from model_metric import ContrastiveModel

# ============================================================
# PARAMETRI DI CONFIGURAZIONE (modifica qui se serve)
# ============================================================
CHECKPOINT_PATH = "checkpoints/best_metric_model.pth"
EXTRACTED_OBJECTS_DIR = "data/extracted_objects"
SPLITS_TO_USE = ["train", "val", "test"]   # su quali split cercare raggruppamenti (tutti = intero dataset)

IMAGE_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 4

SIMILARITY_THRESHOLD = 0.9    # soglia di similarita' coseno per considerare due oggetti "uguali"
                                # (validata sperimentalmente: con AgglomerativeClustering linkage='complete'
                                # produce ~163 cluster coerenti, il piu' numeroso con 15 oggetti)
DBSCAN_MIN_SAMPLES = 2        # minimo di oggetti per formare un cluster (2 = coppie gia' contano)

OUTPUT_DIR = "reports/clustering"
N_CLUSTERS_TO_VISUALIZE = 12   # quanti cluster (i piu' numerosi) visualizzare come griglia immagini
MAX_IMAGES_PER_GRID = 8

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_deterministic_transform(image_size: int):
    """Nessuna randomicita': solo resize e normalizzazione. Ogni oggetto ottiene
    UN embedding stabile, confrontabile in modo coerente con tutti gli altri."""
    return A.Compose([
        A.Resize(image_size, image_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class AllObjectsDataset(Dataset):
    """Carica gli oggetti estratti da piu' split, con transform deterministico."""

    def __init__(self, root_dir, splits, image_size):
        self.samples = []   # lista di (path, split, filename)
        for split in splits:
            split_dir = Path(root_dir) / split
            if not split_dir.exists():
                continue
            for p in sorted(split_dir.iterdir()):
                if p.suffix.lower() in IMG_EXTENSIONS:
                    self.samples.append((p, split))

        self.transform = get_deterministic_transform(image_size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, split = self.samples[idx]
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_t = self.transform(image=image)["image"]
        return {"image": image_t, "filename": path.name, "split": split, "path": str(path)}


def compute_all_embeddings(model, dataset, device):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    embeddings, filenames, splits, paths = [], [], [], []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Calcolo embedding"):
            images = batch["image"].to(device)
            emb = model(images).cpu().numpy()
            embeddings.append(emb)
            filenames.extend(batch["filename"])
            splits.extend(batch["split"])
            paths.extend(batch["path"])

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings, filenames, splits, paths


def run_clustering(embeddings, similarity_threshold, min_samples):
    distance_threshold = 1.0 - similarity_threshold   # distanza coseno = 1 - similarita' coseno
    print(f"\nClustering Agglomerative (linkage='complete', distance_threshold={distance_threshold:.2f})...")

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",   # 'average' = compromesso: la similarita' MEDIA tra i membri deve
                               # superare la soglia. Evita sia il chaining di 'single'/DBSCAN
                               # (gruppi enormi scorrelati) sia la frammentazione eccessiva di
                               # 'complete' (che spacca gruppi validi per una sola coppia sotto soglia)
    )
    labels = clustering.fit_predict(embeddings)

    # con linkage='complete' non esiste il concetto di "rumore" (-1) come in DBSCAN:
    # ogni punto finisce sempre in QUALCHE cluster. Convertiamo in "-1" (nessun match)
    # i cluster che contengono un singolo oggetto, per mantenere la stessa semantica
    # usata nel resto della pipeline (cluster_summary.txt, inference.py, ecc.)
    unique, counts = np.unique(labels, return_counts=True)
    singleton_labels = set(unique[counts < min_samples])
    labels = np.array([-1 if l in singleton_labels else l for l in labels])

    return labels


def save_cluster_grid(cluster_id, paths, output_path: Path, max_images: int):
    selected_paths = paths[:max_images]
    n = len(selected_paths)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 2.8))
    if n == 1:
        axes = [axes]

    for ax, path in zip(axes, selected_paths):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(Path(path).name, fontsize=7)
        ax.axis("off")

    plt.suptitle(f"Cluster {cluster_id} ({len(paths)} oggetti totali, mostrati {n})")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=110)
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint_path = Path(CHECKPOINT_PATH)
    assert checkpoint_path.exists(), f"Checkpoint non trovato: {checkpoint_path.resolve()}"

    model = ContrastiveModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Checkpoint caricato (val_matching_acc: {checkpoint['val_accuracy']:.4f})\n")

    dataset = AllObjectsDataset(EXTRACTED_OBJECTS_DIR, SPLITS_TO_USE, IMAGE_SIZE)
    print(f"Oggetti totali da confrontare: {len(dataset)}\n")

    embeddings, filenames, splits, paths = compute_all_embeddings(model, dataset, device)

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = run_clustering(embeddings, SIMILARITY_THRESHOLD, DBSCAN_MIN_SAMPLES)

    # --- Salva la galleria di embedding su disco (serve per l'inferenza su nuove immagini) ---
    np.save(output_dir / "gallery_embeddings.npy", embeddings)
    with open(output_dir / "gallery_metadata.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split", "path", "cluster_id"])
        for fname, split, path, label in zip(filenames, splits, paths, labels):
            writer.writerow([fname, split, path, label])
    print(f"Galleria embedding salvata in: {output_dir / 'gallery_embeddings.npy'}")

    # --- CSV con l'assegnazione di ogni oggetto al proprio cluster ---
    csv_path = output_dir / "clusters.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split", "cluster_id"])
        for fname, split, label in zip(filenames, splits, labels):
            writer.writerow([fname, split, label])

    # --- Statistiche riassuntive ---
    unique_labels = sorted(set(labels))
    n_noise = int(np.sum(labels == -1))
    real_clusters = [l for l in unique_labels if l != -1]
    cluster_sizes = {l: int(np.sum(labels == l)) for l in real_clusters}
    sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: -x[1])

    summary_lines = [
        "===== RIEPILOGO CLUSTERING =====",
        f"Oggetti totali analizzati: {len(labels)}",
        f"Cluster trovati (oggetti raggruppati con almeno un altro simile): {len(real_clusters)}",
        f"Oggetti SENZA match (istanze uniche nel dataset): {n_noise} ({100*n_noise/len(labels):.1f}%)",
        "",
        "Top 10 cluster piu' numerosi:",
    ]
    for cluster_id, size in sorted_clusters[:10]:
        summary_lines.append(f"   Cluster {cluster_id}: {size} oggetti")

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    with open(output_dir / "cluster_summary.txt", "w") as f:
        f.write(summary_text)

    # --- Griglie visive dei cluster piu' numerosi ---
    for cluster_id, size in sorted_clusters[:N_CLUSTERS_TO_VISUALIZE]:
        cluster_paths = [p for p, l in zip(paths, labels) if l == cluster_id]
        save_cluster_grid(cluster_id, cluster_paths,
                           output_dir / f"cluster_grid_{cluster_id}.png",
                           max_images=MAX_IMAGES_PER_GRID)

    print(f"\nRisultati completi salvati in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()