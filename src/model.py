"""
model.py

U-Net per manipulation localization, usando la libreria
segmentation_models_pytorch (smp):
    - Architettura: U-Net (encoder + decoder, entrambi implementati da smp)
    - Encoder: ResNet18 pretrained su ImageNet, pesi CONGELATI
    - Output: logits grezzi (1 canale), la sigmoid va applicata a parte
      (in inferenza, o implicitamente dentro BCEWithLogitsLoss in training)

Uso rapido (test standalone):
    python model.py
"""

import torch
import segmentation_models_pytorch as smp

# ============================================================
# PARAMETRI DI CONFIGURAZIONE (modifica qui se serve)
# ============================================================
IMAGE_SIZE = 512
ENCODER_NAME = "resnet18"
ENCODER_WEIGHTS = "imagenet"   # None per pesi random
FREEZE_ENCODER = True
TEST_BATCH_SIZE = 2            # solo per il sanity check in fondo al file


def build_model(encoder_name: str = ENCODER_NAME,
                 encoder_weights: str = ENCODER_WEIGHTS,
                 freeze_encoder: bool = FREEZE_ENCODER) -> torch.nn.Module:
    """Costruisce la U-Net smp e, se richiesto, congela l'encoder."""
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,   # None = output logits grezzi, nessuna sigmoid interna
    )

    if freeze_encoder:
        for param in model.encoder.parameters():
            param.requires_grad = False

    model._freeze_encoder = freeze_encoder  # flag usata dall'override di train()

    # Override del metodo train(): se l'encoder e' congelato, lo tiene forzatamente
    # in eval() anche quando si chiama model.train(), per non aggiornare le
    # statistiche di BatchNorm (running_mean/running_var) con i dati del task.
    original_train = model.train

    def train_override(self, mode: bool = True):
        original_train(mode)
        if getattr(self, "_freeze_encoder", False):
            self.encoder.eval()
        return self

    model.train = train_override.__get__(model)

    return model


def _sanity_check():
    """Test standalone: forward pass con tensore random, controllo shape e parametri."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Encoder: {ENCODER_NAME} (weights={ENCODER_WEIGHTS}, freeze={FREEZE_ENCODER})")

    model = build_model().to(device)
    model.train()  # verifica che l'override di train() funzioni correttamente

    dummy_input = torch.randn(TEST_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE).to(device)
    output = model(dummy_input)

    print(f"\nInput shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Output range (logits, pre-sigmoid): [{output.min().item():.3f}, {output.max().item():.3f}]")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"\nParametri totali:     {total_params:,}")
    print(f"Parametri allenabili: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
    print(f"Parametri congelati:  {frozen_params:,} ({100 * frozen_params / total_params:.1f}%)")

    # verifica che l'encoder sia effettivamente in eval mode dopo model.train()
    print(f"\nEncoder in eval mode (atteso True se FREEZE_ENCODER=True): {not model.encoder.training}")


if __name__ == "__main__":
    _sanity_check()