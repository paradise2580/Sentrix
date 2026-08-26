"""
src/models/deep.py

Role
----
Trees see a single snapshot of a seller's current feature values. The LSTM
sees the SEQUENCE — the shape of a build-up over the preceding
`sequence_length` days — which is the pattern the EDA showed exists
(disruption rate moves in waves, not independently day to day).

Two things this module is careful about
---------------------------------------
1. It consumes ALREADY-SCALED features. The sequence frame handed in comes
   from preprocessing.transform_to_frame, so the LSTM sees exactly the same
   representation as the tabular models. Feeding it raw columns would put
   an unscaled `days_since_last_late` of 9999 next to a `late_rate` of
   0.08 and saturate the gates on the first forward pass.

2. It scores EVERY row and reports which rows it scored. Sequences are
   left-padded (see SupplierSequenceDataset) so a seller with three days
   of history still gets a prediction, and the returned index lets the
   evaluator line every model up on identical rows. Without that, PR-AUC
   values computed over different row sets end up printed in one ranked
   table, and the ranking means nothing.
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

TARGET = load_config()["model"]["target_column"]


class SupplierSequenceDataset(Dataset):
    """
    Fixed-length sliding-window sequences per seller, so the LSTM sees a
    (seq_len, n_features) tensor per sample instead of a single row.

    Sequences are LEFT-PADDED
    -------------------------
    A seller's day 3 has only two prior days of history, not thirty. The
    obvious handling — skip any row without a full window — has two costs
    that are easy to miss:

    1. It silently shrinks the evaluation set. Dropping every seller's
       first 30 days removed 39% of the test block here (76,815 rows down
       to 46,715), and the rows it removed were not a random sample:
       early seller-days carry a higher late rate, so the surviving base
       rate fell from 16.8% to 15.0%. Every model was then compared on
       that smaller, easier subset.
    2. It cannot score a new seller at all, which is precisely when a
       delivery-risk score is most useful.

    So short windows are padded at the FRONT with zeros. The features are
    already standardised when they reach here, so a zero row is the
    training mean — i.e. "no information", which is exactly what an
    unobserved day is. Every row in every block gets a prediction, and the
    evaluation set is the whole block.

    Exposes `row_index`: the DataFrame index label of the row each sample
    PREDICTS (the row at the end of the window), so predictions can be
    joined back to the source frame.
    """

    def __init__(self, df: pd.DataFrame, feature_cols: list[str], seq_len: int):
        self.seq_len = seq_len
        n_features = len(feature_cols)
        samples, labels, row_index = [], [], []

        df = df.sort_values(["seller_id", "as_of_date"])
        for _, group in df.groupby("seller_id", sort=True):
            values = group[feature_cols].fillna(0).values.astype(np.float32)
            y = group[TARGET].values.astype(np.float32)
            idx = group.index.values

            for i in range(len(group)):
                window = values[max(0, i - seq_len):i]
                if len(window) < seq_len:
                    pad = np.zeros((seq_len - len(window), n_features), dtype=np.float32)
                    window = np.vstack([pad, window]) if len(window) else pad
                samples.append(window)
                labels.append(y[i])
                row_index.append(idx[i])

        self.samples = (np.stack(samples) if samples
                        else np.empty((0, seq_len, len(feature_cols)), dtype=np.float32))
        self.labels = np.array(labels, dtype=np.float32)
        self.row_index = np.array(row_index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.samples[idx]), torch.tensor(self.labels[idx])


class DisruptionLSTM(nn.Module):
    """A small LSTM classifier: sequence in, disruption-probability logit out."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int,
                 dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1)
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]                              # (batch, hidden_size)
        return self.classifier(last_hidden).squeeze(-1)    # raw logits


def train_lstm(seq_df: pd.DataFrame, feature_cols: list[str]) -> tuple[DisruptionLSTM, dict]:
    """
    Train the LSTM on sliding-window sequences.

    seq_df : scaled frame from preprocessing.transform_to_frame, containing
             the feature columns plus seller_id, as_of_date and the label.

    Returns the trained model and the metadata needed to rebuild identical
    sequences at inference time.
    """
    try:
        full_cfg = load_config()
        cfg = full_cfg["model"]["lstm"]
        seed = full_cfg["project"]["random_state"]
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset = SupplierSequenceDataset(seq_df, feature_cols, cfg["sequence_length"])
        if len(dataset) == 0:
            raise ValueError("The training frame is empty — no sequences to build.")
        logger.info(f"LSTM dataset: {len(dataset):,} sequences of length "
                    f"{cfg['sequence_length']} x {len(feature_cols)} features")

        loader = TorchDataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True)

        model = DisruptionLSTM(len(feature_cols), cfg["hidden_size"],
                               cfg["num_layers"], cfg.get("dropout", 0.0))
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

        # pos_weight compensates for class imbalance inside the loss itself,
        # which is the sequence-model equivalent of class_weight="balanced".
        # SMOTE is not an option here: interpolating between two sequences
        # produces a trajectory no seller ever had.
        n_pos = float(dataset.labels.sum())
        n_neg = float(len(dataset.labels) - n_pos)
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        model.train()
        for epoch in range(cfg["epochs"]):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item() * len(y_batch)
            logger.info(f"LSTM epoch {epoch + 1}/{cfg['epochs']} — "
                        f"loss: {epoch_loss / len(dataset):.4f}")

        return model, {"feature_cols": list(feature_cols),
                       "seq_len": cfg["sequence_length"],
                       "hidden_size": cfg["hidden_size"],
                       "num_layers": cfg["num_layers"]}
    except Exception as e:
        raise SentrixException(e, sys)


@torch.no_grad()
def predict_lstm(model: DisruptionLSTM, seq_df: pd.DataFrame,
                 meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference over a scaled sequence frame.

    Returns
    -------
    (probs, row_index)
        probs      predicted probability per scored row
        row_index  the DataFrame index labels those probabilities belong
                   to — the caller uses this to align every other model
                   onto the same evaluation rows.
    """
    try:
        model.eval()
        dataset = SupplierSequenceDataset(seq_df, meta["feature_cols"], meta["seq_len"])
        if len(dataset) == 0:
            return np.array([]), np.array([])

        loader = TorchDataLoader(dataset, batch_size=512, shuffle=False)
        probs = [torch.sigmoid(model(X_batch)).numpy() for X_batch, _ in loader]
        return np.concatenate(probs), dataset.row_index
    except Exception as e:
        raise SentrixException(e, sys)
