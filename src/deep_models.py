"""Deep learning models for satellite image time-series (SITS) classification.

Implements:
- TempCNN  : 1D temporal convolutional network (Pelletier et al., 2019)
- TransformerSITS : self-attention over the S1/S2 time-series
- PrithviHead : fine-tuning stub for the NASA/IBM Prithvi geospatial
  foundation model (HLS) embeddings.

Input tensor convention: (batch, time, channels) for sequence models.
"""
import math
import torch
import torch.nn as nn


class TempCNN(nn.Module):
    """Temporal CNN for SITS pixel classification.

    Args:
        n_channels: number of spectral/SAR bands per timestep.
        n_classes: output classes (wheat / non-wheat = 2).
        seq_len: number of timesteps in the season.
    """

    def __init__(self, n_channels=8, n_classes=2, seq_len=12, hidden=64,
                 dropout=0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, hidden, 5, padding=2), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(hidden * seq_len, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):  # x: (B, T, C)
        x = x.transpose(1, 2)            # -> (B, C, T)
        return self.head(self.conv(x))


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerSITS(nn.Module):
    """Self-attention classifier over the satellite time-series.

    Captures long-range temporal dependencies across the Rabi phenology
    (sowing -> tillering -> heading -> maturity) that a windowed RF misses.
    """

    def __init__(self, n_channels=8, n_classes=2, d_model=64, nhead=4,
                 nlayers=3, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(n_channels, d_model)
        self.posenc = _PositionalEncoding(d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                         dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, nlayers)
        self.cls = nn.Sequential(nn.LayerNorm(d_model),
                                 nn.Linear(d_model, n_classes))

    def forward(self, x):  # x: (B, T, C)
        h = self.posenc(self.embed(x))
        h = self.encoder(h)
        return self.cls(h.mean(dim=1))   # temporal mean pooling


class PrithviHead(nn.Module):
    """Lightweight classification head on top of frozen Prithvi embeddings.

    Load Prithvi from Hugging Face ('ibm-nasa-geospatial/Prithvi-100M'),
    extract patch embeddings, freeze the backbone, and train this head.
    Backbone loading is left to the notebook to avoid a heavy import here.
    """

    def __init__(self, embed_dim=768, n_classes=2, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, 256), nn.GELU(),
            nn.Linear(256, n_classes))

    def forward(self, embeddings):  # (B, embed_dim)
        return self.head(embeddings)


def train_epoch(model, loader, optimizer, criterion, device="cpu"):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device="cpu"):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return correct / max(total, 1)
