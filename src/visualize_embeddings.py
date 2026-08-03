"""
visualize_embeddings.py

Visualizza la qualita' dello spazio di embedding imparato dal modello
di metric learning, con due grafici complementari:

    1. Scatter t-SNE: riduce gli embedding (128-dim) a 2D. Per un
       sottoinsieme di oggetti, genera piu' view augmentate e le colora
       allo stesso modo -> se il training ha funzionato, le view dello
       stesso oggetto formano piccoli gruppi compatti, ben separati tra loro.

    2. Istogramma delle similarita' coseno: confronta la distribuzione
       delle similarita' tra COPPIE VERE (stesso oggetto, 2 view diverse)
       e COPPIE CASUALI (oggetti diversi). Meno le due distribuzioni si
       sovrappongono, meglio l'embedding separa istanze diverse.

Uso:
    python visualize_embeddings.py
    (modifica i parametri in cima al file se serve)
"""

from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from dataset_metric import ContrastiveObjectDataset, IMAGE_SIZE
from model_metric import ContrastiveModel

# ============================================================
# PARAMETRI DI CONFIGURAZIONE (modifica qui se serve)
# ============================================================
CHECKPOINT_PATH = "checkpoints/best_metric_model.pth"
EXTRACTED_OBJECTS_DIR = "data/extracted_objects"
SPLIT = "val"

N_OBJECTS_TSNE = 30           # numero di oggetti distinti nello scatter (troppi = illeggibile)
N_VIEWS_PER_OBJECT = 5        # quante view augmentate per oggetto, nello scatter

N_OBJECTS_HISTOGRAM = 300     # numero di oggetti usati per calcolare le distribuzioni di similarita'

OUTPUT_DIR = "reports/eval"
SEED = 42


def load_model(device):
    checkpoint_path = Path(CHECKPOINT_PATH)
    assert checkpoint_path.exists(), f"Checkpoint non trovato: {checkpoint_path.resolve()}"

    model = ContrastiveModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Checkpoint caricato: {checkpoint_path} (val_matching_acc: {checkpoint['val_accuracy']:.4f})\n")
    return model


@torch.no_grad()
def compute_embedding(model, dataset, idx, device):
    """Genera UNA view augmentata casuale dell'oggetto idx e ne calcola l'embedding."""
    sample = dataset[idx]
    view = sample["view_a"].unsqueeze(0).to(device)
    embedding = model(view)[0].cpu().numpy()
    return embedding, sample["filename"]


def plot_tsne_scatter(model, dataset, device, output_path: Path, n_objects: int, n_views: int, seed: int):
    rng = np.random.default_rng(seed)
    object_indices = rng.choice(len(dataset), size=min(n_objects, len(dataset)), replace=False)

    embeddings = []
    object_ids = []
    filenames = []

    for obj_id, idx in enumerate(object_indices):
        for _ in range(n_views):
            emb, fname = compute_embedding(model, dataset, idx, device)
            embeddings.append(emb)
            object_ids.append(obj_id)
            filenames.append(fname)

    embeddings = np.array(embeddings)
    object_ids = np.array(object_ids)

    n_samples = len(embeddings)
    perplexity = min(30, max(5, n_samples // 4))

    print(f"Calcolo t-SNE su {n_samples} embedding ({n_objects} oggetti x {n_views} view, perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca")
    coords_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(9, 8))
    cmap = plt.get_cmap("tab20" if n_objects <= 20 else "gist_ncar")
    colors = [cmap(i / n_objects) for i in object_ids]

    plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors, s=60, alpha=0.8, edgecolors="black", linewidths=0.3)
    plt.title(f"t-SNE dello spazio di embedding\n({n_objects} oggetti, {n_views} view augmentate ciascuno, stesso colore = stesso oggetto)")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=130)
    plt.close()
    print(f"Scatter t-SNE salvato in: {output_path.resolve()}")


@torch.no_grad()
def plot_similarity_histogram(model, dataset, device, output_path: Path, n_objects: int, seed: int):
    rng = np.random.default_rng(seed)
    object_indices = rng.choice(len(dataset), size=min(n_objects, len(dataset)), replace=False)

    # per ogni oggetto, calcola l'embedding di 2 view diverse (coppia "vera")
    embeddings_a, embeddings_b = [], []
    for idx in object_indices:
        sample = dataset[idx]
        view_a = sample["view_a"].unsqueeze(0).to(device)
        view_b = sample["view_b"].unsqueeze(0).to(device)
        emb_a = model(view_a)[0].cpu().numpy()
        emb_b = model(view_b)[0].cpu().numpy()
        embeddings_a.append(emb_a)
        embeddings_b.append(emb_b)

    embeddings_a = np.array(embeddings_a)   # (N, D)
    embeddings_b = np.array(embeddings_b)   # (N, D)

    # similarita' coseno delle COPPIE VERE (stesso oggetto, view A vs view B)
    positive_sims = np.sum(embeddings_a * embeddings_b, axis=1)   # gia' normalizzati L2 -> dot product = cosine sim

    # similarita' coseno di COPPIE CASUALI (view A di un oggetto vs view A di un ALTRO oggetto)
    negative_sims = []
    n = len(embeddings_a)
    rng2 = np.random.default_rng(seed + 1)
    for i in range(n):
        j = rng2.integers(0, n)
        while j == i:
            j = rng2.integers(0, n)
        negative_sims.append(float(np.dot(embeddings_a[i], embeddings_a[j])))
    negative_sims = np.array(negative_sims)

    plt.figure(figsize=(8, 5))
    plt.hist(positive_sims, bins=40, alpha=0.6, label="Coppie vere (stesso oggetto)", color="seagreen")
    plt.hist(negative_sims, bins=40, alpha=0.6, label="Coppie casuali (oggetti diversi)", color="indianred")
    plt.xlabel("Similarita' coseno")
    plt.ylabel("Numero di coppie")
    plt.title("Distribuzione delle similarita': coppie vere vs casuali")
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=130)
    plt.close()
    print(f"Istogramma similarita' salvato in: {output_path.resolve()}")

    print(f"\nSimilarita' media coppie VERE:    {positive_sims.mean():.4f} (std {positive_sims.std():.4f})")
    print(f"Similarita' media coppie CASUALI: {negative_sims.mean():.4f} (std {negative_sims.std():.4f})")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(device)
    dataset = ContrastiveObjectDataset(EXTRACTED_OBJECTS_DIR, split=SPLIT, image_size=IMAGE_SIZE)
    print(f"Dataset '{SPLIT}': {len(dataset)} oggetti\n")

    output_dir = Path(OUTPUT_DIR)

    plot_tsne_scatter(model, dataset, device, output_dir / "embedding_tsne.png",
                       n_objects=N_OBJECTS_TSNE, n_views=N_VIEWS_PER_OBJECT, seed=SEED)

    plot_similarity_histogram(model, dataset, device, output_dir / "embedding_similarity_histogram.png",
                               n_objects=N_OBJECTS_HISTOGRAM, seed=SEED)


if __name__ == "__main__":
    main()