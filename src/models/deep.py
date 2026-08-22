"""
src/models/deep.py

Role
----
Trees see a single snapshot of a supplier's current feature values. The
LSTM sees the SEQUENCE — the shape of a build-up over the preceding
sequence_length days — which is exactly the pattern Phase 3's EDA
insight 3 showed exists (disruption rate moves in waves, not randomly).

This module builds a per-supplier sliding-window sequence dataset from
the feature table and trains a small LSTM classifier over it in PyTorch.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from src.config_loader import load_config
from src.logger import get_logger
from src.exception import SentrixException
import sys

logger = get_logger(__name__)


class SupplierSequenceDataset(Dataset):
    """
    Builds fixed-length sliding-window sequences per supplier from the
    feature table, so the LSTM sees a (sequence_length, n_features) tensor
    per sample instead of a single row.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list[str], seq_len: int):
        self.seq_len = seq_len
        self.samples = []
        self.labels = []

        df = df.sort_values(["seller_id", "as_of_date"])
        for _, group in df.groupby("seller_id"):
            values = group[feature_cols].fillna(0).values.astype(np.float32)
            labels = group["disruption_next_30d"].values.astype(np.float32)

            for i in range(seq_len, len(group)):
                self.samples.append(values[i - seq_len:i])
                self.labels.append(labels[i])

        self.samples = np.stack(self.samples) if self.samples else np.empty((0, seq_len, len(feature_cols)))
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return torch.tensor(self.samples[idx]), torch.tensor(self.labels[idx])


class DisruptionLSTM(nn.Module):
    """A small LSTM classifier: sequence in, disruption-probability logit out."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]                 # (batch, hidden_size)
        return self.classifier(last_hidden).squeeze(-1)   # raw logits


def train_lstm(df: pd.DataFrame, feature_cols: list[str]) -> tuple[DisruptionLSTM, dict]:
    """
    Train the LSTM on sliding-window sequences. Returns the trained model
    and a small metadata dict (feature_cols, seq_len) needed to build
    identical sequences again at inference time.
    """
    try:
        cfg = load_config()["model"]["lstm"]
        seed = load_config()["project"]["random_state"]
        torch.manual_seed(seed)

        dataset = SupplierSequenceDataset(df, feature_cols, cfg["sequence_length"])
        logger.info(f"LSTM dataset built: {len(dataset)} sequences of length {cfg['sequence_length']}")

        loader = TorchDataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

        model = DisruptionLSTM(len(feature_cols), cfg["hidden_size"], cfg["num_layers"])
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

        # pos_weight compensates for class imbalance in the loss itself
        n_pos = dataset.labels.sum()
        n_neg = len(dataset.labels) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model.train()
        for epoch in range(cfg["epochs"]):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(y_batch)
            epoch_loss /= len(dataset)
            logger.info(f"LSTM epoch {epoch + 1}/{cfg['epochs']} — loss: {epoch_loss:.4f}")

        return model, {"feature_cols": feature_cols, "seq_len": cfg["sequence_length"]}
    except Exception as e:
        raise SentrixException(e, sys)


@torch.no_grad()
def predict_lstm(model: DisruptionLSTM, df: pd.DataFrame, meta: dict) -> np.ndarray:
    """Run inference, returning disruption probabilities for every valid sequence."""
    try:
        model.eval()
        dataset = SupplierSequenceDataset(df, meta["feature_cols"], meta["seq_len"])
        if len(dataset) == 0:
            return np.array([])
        loader = TorchDataLoader(dataset, batch_size=256, shuffle=False)

        probs = []
        for X_batch, _ in loader:
            logits = model(X_batch)
            probs.append(torch.sigmoid(logits).numpy())
        return np.concatenate(probs)
    except Exception as e:
        raise SentrixException(e, sys)
