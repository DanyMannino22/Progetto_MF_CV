# Multimedia Forensics & Computer Vision — Manipulation Localization, Object Extraction & Instance Matching

Pipeline completa per la rilevazione di manipolazioni in immagini digitali:
1. **Segmentazione** dell'area manipolata (U-Net, encoder ResNet18 pretrained)
2. **Estrazione** dell'oggetto manipolato (bounding box dalla maschera)
3. **Metric learning** per riconoscere se due oggetti estratti sono la stessa istanza (contrastive learning self-supervised, NT-Xent)
4. **Inferenza end-to-end** su nuove immagini

Relazione tecnica completa disponibile in [`relazione_progetto.docx`](./relazione_progetto.docx).

## Struttura del repository

```
src/
├── dataset.py                              Dataset PyTorch per la segmentazione
├── model.py                                U-Net (segmentation_models_pytorch, encoder ResNet18)
├── train.py                                Training con encoder congelato
├── finetune.py                             Fine-tuning con encoder sbloccato (LR differenziato)
├── evaluate.py                             Valutazione finale sul test set
├── extract_objects.py                      Estrazione oggetti da maschere ground truth
├── extract_objects_from_predictions.py     Estrazione oggetti da maschere predette (pipeline reale)
├── dataset_metric.py                       Dataset per il metric learning (view augmentate)
├── model_metric.py                         Rete Siamese (ResNet18 + projection head)
├── train_metric.py                         Training contrastive (NT-Xent loss)
├── visualize_embeddings.py                 Validazione qualitativa (t-SNE, istogramma similarità)
├── cluster_objects.py                      Clustering degli oggetti sull'intero dataset + galleria
└── inference.py                            Pipeline end-to-end su nuove immagini
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# PyTorch con supporto CUDA (verifica la versione corretta su pytorch.org in base alla tua GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt
```

## Dataset

Il progetto si aspetta la seguente struttura in `data/` (non inclusa nel repository per dimensione):

```
data/
├── train/{images,masks}/
├── val/{images,masks}/
└── test/{images,masks}/
```

## Pipeline — ordine di esecuzione

```bash
# 1. Segmentazione: training con encoder congelato
python src/train.py

# 2. Segmentazione: fine-tuning con encoder sbloccato
python src/finetune.py

# 3. Valutazione sul test set
python src/evaluate.py

# 4. Estrazione oggetti (da ground truth, per costruire il dataset del metric learning)
python src/extract_objects.py

# 5. (Opzionale) Estrazione da maschere predette, per validare la pipeline end-to-end
python src/extract_objects_from_predictions.py

# 6. Metric learning: training contrastive
python src/train_metric.py

# 7. Validazione qualitativa dell'embedding
python src/visualize_embeddings.py

# 8. Clustering sull'intero dataset + generazione della galleria di riferimento
python src/cluster_objects.py

# 9. Inferenza su una nuova immagine
python src/inference.py --image "percorso/alla/immagine.jpg"
```

Monitoraggio del training via TensorBoard:
```bash
tensorboard --logdir runs
```

## Risultati principali

| Fase | Metrica | Valore |
|---|---|---|
| Segmentazione (test set) | Dice | 0.5266 |
| Segmentazione (test set) | IoU | 0.4496 |
| Estrazione oggetti (da predizioni) | Copertura | ~97-98% |
| Metric learning | Similarità coppie vere | 0.937 ± 0.064 |
| Metric learning | Similarità coppie casuali | -0.0005 ± 0.121 |
| Clustering | Cluster coerenti trovati | 163 |

Dettagli completi, scelte metodologiche, problemi incontrati (overfitting nel fine-tuning, chaining nel clustering) e come sono stati risolti: vedi la relazione tecnica allegata.

## Note tecniche

- L'encoder della U-Net viene congelato in una prima fase (incluso il freeze delle statistiche BatchNorm), poi sbloccato per il fine-tuning con learning rate differenziato (encoder 1e-5, decoder 1e-4).
- Il metric learning non richiede etichette di classe: tratta ogni oggetto estratto come la propria istanza unica (instance discrimination), con augmentation pensate per simulare le trasformazioni realistiche di un oggetto riutilizzato in una manipolazione (resize, compressione JPEG, variazioni di colore).
- La soglia di similarità per il matching (0.85-0.9) è stata tarata empiricamente sulla distribuzione di similarità tra coppie vere e casuali.