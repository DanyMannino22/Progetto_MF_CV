"""
model_metric.py

Modello per instance discrimination (metric learning self-supervised):
    - Encoder: ResNet18 pretrained ImageNet (pesi FRESCHI, indipendenti
      dal modello di segmentazione)
    - Projection head: piccolo MLP (come in SimCLR), mappa le feature
      dell'encoder in uno spazio di embedding a bassa dimensionalita'
    - L'architettura e' "Siamese": la STESSA rete (pesi condivisi) viene
      applicata a entrambe le view augmentate di un oggetto

Uso rapido (test standalone):
    python model_metric.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

# ============================================================
# PARAMETRI DI CONFIGURAZIONE (modifica qui se serve)
# ============================================================
EMBEDDING_DIM = 128          # dimensione finale dell'embedding (dopo projection head)
PROJECTION_HIDDEN_DIM = 512  # dimensione nascosta del MLP della projection head
PRETRAINED = True
IMAGE_SIZE = 224
TEST_BATCH_SIZE = 4


class ProjectionHead(nn.Module):
    """MLP a due layer, come in SimCLR: Linear -> BN -> ReLU -> Linear."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ContrastiveModel(nn.Module):
    def __init__(self, embedding_dim: int = EMBEDDING_DIM,
                 hidden_dim: int = PROJECTION_HIDDEN_DIM,
                 pretrained: bool = PRETRAINED):
        super().__init__()

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        encoder_out_dim = backbone.fc.in_features   # 512 per ResNet18

        backbone.fc = nn.Identity()   # rimuove il classificatore ImageNet, teniamo solo le feature
        self.encoder = backbone

        self.projection_head = ProjectionHead(encoder_out_dim, hidden_dim, embedding_dim)

    def forward(self, x):
        features = self.encoder(x)                       # (B, 512)
        embedding = self.projection_head(features)        # (B, embedding_dim)
        embedding = F.normalize(embedding, dim=1)          # L2-normalizzazione: fondamentale per NT-Xent
        return embedding

    def encode(self, x):
        """Ritorna solo le feature dell'encoder (pre-projection head), utili in
        inferenza: in molti lavori di metric learning si usano le feature
        dell'encoder (piu' generali) invece dell'embedding della projection head
        (che e' ottimizzato specificamente per la loss contrastive)."""
        with torch.no_grad():
            return self.encoder(x)


def _sanity_check():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ContrastiveModel().to(device)
    model.train()

    dummy_a = torch.randn(TEST_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    dummy_b = torch.randn(TEST_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)

    emb_a = model(dummy_a)
    emb_b = model(dummy_b)

    print(f"Embedding A shape: {tuple(emb_a.shape)}")
    print(f"Embedding B shape: {tuple(emb_b.shape)}")
    print(f"Norma L2 embedding A (attesa ~1.0): {emb_a.norm(dim=1).mean().item():.4f}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParametri totali: {total_params:,}")


if __name__ == "__main__":
    _sanity_check()