"""
explore_dataset.py

Esplora il dataset di immagini manipolate + maschere di ground truth,
producendo statistiche e visualizzazioni utili per progettare il modello
di segmentazione (dimensione input, loss function, gestione class imbalance).

Genera in --output_dir:
    - stats_<split>.csv          statistiche per-immagine
    - summary_<split>.txt        riepilogo aggregato leggibile
    - white_ratio_hist_<split>.png   istogramma % pixel manipolati
    - image_sizes_<split>.png        distribuzione dimensioni immagini
    - sample_grid_<split>.png        griglia di campioni casuali (img | mask | overlay)

Uso:
    python explore_dataset.py --data_dir data --output_dir reports/eda --splits train val test
"""

import argparse
import csv
from pathlib import Path
from collections import Counter

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # backend non interattivo, per salvare su file senza display
import matplotlib.pyplot as plt
from tqdm import tqdm

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_files(directory: Path):
    return sorted([p for p in directory.iterdir() if p.suffix.lower() in IMG_EXTENSIONS])


def analyze_split(split_dir: Path):
    """Calcola statistiche per ogni coppia immagine/maschera in uno split."""
    images_dir = split_dir / "images"
    masks_dir = split_dir / "masks"

    image_files = list_files(images_dir)
    mask_files = {p.stem: p for p in list_files(masks_dir)}

    records = []
    skipped = []

    for img_path in tqdm(image_files, desc=f"Analisi {split_dir.name}"):
        mask_path = mask_files.get(img_path.stem)
        if mask_path is None:
            skipped.append(img_path.name)
            continue

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            skipped.append(img_path.name)
            continue

        img_h, img_w = img.shape[:2]
        mask_h, mask_w = mask.shape[:2]

        white_pixels = int(np.sum(mask > 127))
        total_pixels = mask.shape[0] * mask.shape[1]
        white_ratio = white_pixels / total_pixels

        unique_vals = np.unique(mask)
        is_strictly_binary = set(unique_vals.tolist()).issubset({0, 255})

        records.append({
            "filename": img_path.name,
            "img_width": img_w,
            "img_height": img_h,
            "mask_width": mask_w,
            "mask_height": mask_h,
            "dims_match": (img_w == mask_w) and (img_h == mask_h),
            "white_ratio": round(white_ratio, 6),
            "n_unique_mask_values": len(unique_vals),
            "is_strictly_binary": is_strictly_binary,
        })

    return records, skipped


def save_csv(records, output_path: Path):
    if not records:
        return
    keys = records[0].keys()
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def save_summary(records, skipped, split_name, output_path: Path):
    white_ratios = np.array([r["white_ratio"] for r in records])
    sizes = Counter((r["img_width"], r["img_height"]) for r in records)
    dims_mismatch = sum(1 for r in records if not r["dims_match"])
    non_binary = sum(1 for r in records if not r["is_strictly_binary"])
    empty_masks = int(np.sum(white_ratios == 0.0))

    lines = [
        f"===== SUMMARY: {split_name} =====",
        f"Coppie analizzate: {len(records)}",
        f"Coppie scartate (mancanti/corrotte): {len(skipped)}",
        "",
        "--- Dimensioni immagini ---",
        f"Dimensioni uniche trovate: {len(sizes)}",
        "Top 5 dimensioni piu' comuni:",
    ]
    for (w, h), count in sizes.most_common(5):
        lines.append(f"   {w}x{h}: {count} immagini")

    lines += [
        "",
        f"Coppie con dimensione img != maschera: {dims_mismatch}",
        f"Maschere NON strettamente binarie (antialiasing/grigi): {non_binary}",
        "",
        "--- Area manipolata (white_ratio = pixel bianchi / totale) ---",
        f"Media:   {white_ratios.mean():.4f}",
        f"Mediana: {np.median(white_ratios):.4f}",
        f"Min:     {white_ratios.min():.4f}",
        f"Max:     {white_ratios.max():.4f}",
        f"Std:     {white_ratios.std():.4f}",
        f"Maschere completamente vuote (white_ratio == 0): {empty_masks}",
    ]

    if skipped:
        lines += ["", f"File scartati (prime 10 di {len(skipped)}):"]
        lines += [f"   - {name}" for name in skipped[:10]]

    text = "\n".join(lines)
    print("\n" + text + "\n")
    with open(output_path, "w") as f:
        f.write(text)


def plot_white_ratio_hist(records, split_name, output_path: Path):
    white_ratios = [r["white_ratio"] for r in records]
    plt.figure(figsize=(8, 5))
    plt.hist(white_ratios, bins=50, color="steelblue", edgecolor="black")
    plt.xlabel("Frazione di pixel manipolati (white_ratio)")
    plt.ylabel("Numero di immagini")
    plt.title(f"Distribuzione area manipolata - {split_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def plot_image_sizes(records, split_name, output_path: Path):
    widths = [r["img_width"] for r in records]
    heights = [r["img_height"] for r in records]
    plt.figure(figsize=(6, 6))
    plt.scatter(widths, heights, alpha=0.4, s=10, color="darkorange")
    plt.xlabel("Larghezza (px)")
    plt.ylabel("Altezza (px)")
    plt.title(f"Distribuzione dimensioni immagini - {split_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def save_sample_grid(split_dir: Path, records, split_name, output_path: Path, n_samples=6, seed=42):
    """Griglia: immagine | maschera | overlay, per n_samples campioni casuali."""
    rng = np.random.default_rng(seed)
    if len(records) == 0:
        return
    chosen = rng.choice(records, size=min(n_samples, len(records)), replace=False)

    fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 3 * len(chosen)))
    if len(chosen) == 1:
        axes = axes.reshape(1, -1)

    for i, rec in enumerate(chosen):
        img_path = split_dir / "images" / rec["filename"]
        mask_path_candidates = list((split_dir / "masks").glob(f"{Path(rec['filename']).stem}.*"))
        mask_path = mask_path_candidates[0] if mask_path_candidates else None

        img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        overlay = img.copy()
        red_layer = np.zeros_like(img)
        red_layer[..., 0] = 255
        mask_bool = mask > 127
        overlay[mask_bool] = cv2.addWeighted(img, 0.5, red_layer, 0.5, 0)[mask_bool]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(rec["filename"], fontsize=8)
        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("mask", fontsize=8)
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title(f"overlay (wr={rec['white_ratio']:.3f})", fontsize=8)

        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle(f"Campioni casuali - {split_name}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Esplorazione dataset immagini/maschere")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="reports/eda")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=6, help="Numero campioni nella griglia visiva")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in args.splits:
        split_dir = data_dir / split_name
        if not split_dir.exists():
            print(f"[SKIP] Cartella non trovata: {split_dir}")
            continue

        records, skipped = analyze_split(split_dir)

        if not records:
            print(f"[SKIP] Nessuna coppia valida trovata in {split_dir}")
            continue

        save_csv(records, output_dir / f"stats_{split_name}.csv")
        save_summary(records, skipped, split_name, output_dir / f"summary_{split_name}.txt")
        plot_white_ratio_hist(records, split_name, output_dir / f"white_ratio_hist_{split_name}.png")
        plot_image_sizes(records, split_name, output_dir / f"image_sizes_{split_name}.png")
        save_sample_grid(split_dir, records, split_name,
                          output_dir / f"sample_grid_{split_name}.png", n_samples=args.n_samples)

    print(f"\nReport completo salvato in: {output_dir.resolve()}")


if __name__ == "__main__":
    main()