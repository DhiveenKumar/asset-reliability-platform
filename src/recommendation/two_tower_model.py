# =============================================================================
# two_tower_model.py — Two-Tower Embedding Recommendation Model
#
# Learns dense vector embeddings for failure modes and maintenance
# actions jointly, such that historically co-occurring pairs end up
# close together in embedding space. Directly comparable to
# industry-standard two-tower recommendation architectures.
# =============================================================================

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CooccurrenceDataset(Dataset):
    """
    Creates training pairs from historical co-occurrence: for every
    (failure_mode, action) pair that actually appeared together in
    a historical work order, that's a POSITIVE example (label 1).
    We also generate an equal number of random NEGATIVE pairs
    (failure mode + action that never co-occurred) labeled 0 -
    the model needs both positive and negative examples to learn
    what "similar" versus "dissimilar" actually means.
    """
    def __init__(self, df: pd.DataFrame, failure_modes: list, actions: list):
        self.failure_to_idx = {fm: i for i, fm in enumerate(failure_modes)}
        self.action_to_idx = {a: i for i, a in enumerate(actions)}

        positive_pairs = set(
            (row["failure_mode"], row["action"]) for _, row in df.iterrows()
        )

        samples = []
        for fm, action in positive_pairs:
            samples.append((self.failure_to_idx[fm], self.action_to_idx[action], 1.0))

        np.random.seed(42)
        n_negative = len(positive_pairs)
        for _ in range(n_negative):
            fm_idx = np.random.randint(len(failure_modes))
            action_idx = np.random.randint(len(actions))
            fm = failure_modes[fm_idx]
            action = actions[action_idx]
            if (fm, action) not in positive_pairs:
                samples.append((fm_idx, action_idx, 0.0))

        self.samples = samples
        logger.info(f"Created {len(self.samples)} training pairs "
                    f"({len(positive_pairs)} positive, "
                    f"{len(self.samples) - len(positive_pairs)} negative)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fm_idx, action_idx, label = self.samples[idx]
        return (
            torch.tensor(fm_idx, dtype=torch.long),
            torch.tensor(action_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )


class TwoTowerModel(nn.Module):
    """
    Two separate embedding "towers" - one learns an embedding table
    for failure modes, the other for actions. Both map into the SAME
    dimensional space (embedding_dim), so their vectors can be
    directly compared via dot product.

    This is deliberately simple (just embedding lookups, no deep
    layers) because our vocabulary is small (8 failure modes, 23
    actions) - a real production system with millions of items would
    typically add additional dense layers per tower to incorporate
    richer features beyond just an ID lookup.
    """
    def __init__(self, n_failure_modes: int, n_actions: int, embedding_dim: int = 16):
        super().__init__()
        self.failure_tower = nn.Embedding(n_failure_modes, embedding_dim)
        self.action_tower = nn.Embedding(n_actions, embedding_dim)

    def forward(self, failure_idx, action_idx):
        failure_vec = self.failure_tower(failure_idx)
        action_vec = self.action_tower(action_idx)
        # Dot product similarity - higher value means the model
        # believes these are more strongly associated
        score = (failure_vec * action_vec).sum(dim=1)
        return torch.sigmoid(score)


def train_two_tower(dataset, n_failure_modes, n_actions, epochs=50):
    model = TwoTowerModel(n_failure_modes, n_actions)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    criterion = nn.BCELoss()

    logger.info("Training two-tower model...")

    for epoch in range(epochs):
        total_loss = 0
        for fm_idx, action_idx, label in loader:
            optimizer.zero_grad()
            pred = model(fm_idx, action_idx)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    return model


def recommend_with_two_tower(
    model, failure_mode: str, failure_modes: list, actions: list, top_k: int = 5
) -> list:
    failure_idx = torch.tensor([failure_modes.index(failure_mode)])
    failure_vec = model.failure_tower(failure_idx).detach().numpy()

    action_vecs = model.action_tower.weight.detach().numpy()

    similarities = cosine_similarity(failure_vec, action_vecs)[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    return [
        {"action": actions[i], "score": round(float(similarities[i]), 3)}
        for i in top_indices
    ]


def evaluate_precision_at_k(model, df, failure_modes, actions, k=5):
    """
    Precision@K: of the top K recommendations, what fraction were
    ACTUALLY historically associated with that failure mode?
    This is the standard evaluation metric for recommendation
    systems - different from classification accuracy because
    ranking quality matters, not just binary correctness.
    """
    precisions = []

    for fm in failure_modes:
        actual_actions = set(df[df["failure_mode"] == fm]["action"].unique())
        recs = recommend_with_two_tower(model, fm, failure_modes, actions, top_k=k)
        recommended_actions = set(r["action"] for r in recs)

        hits = len(recommended_actions & actual_actions)
        precision = hits / k
        precisions.append(precision)

    avg_precision = np.mean(precisions)
    logger.info(f"Precision@{k}: {avg_precision:.3f}")

    return avg_precision


def run_two_tower_training():
    logger.info("=" * 60)
    logger.info("Two-Tower Recommendation Model")
    logger.info("=" * 60)

    df = pd.read_csv("data/recommendation/work_order_actions.csv")

    failure_modes = sorted(df["failure_mode"].unique().tolist())
    actions = sorted(df["action"].unique().tolist())

    dataset = CooccurrenceDataset(df, failure_modes, actions)
    model = train_two_tower(dataset, len(failure_modes), len(actions))

    precision = evaluate_precision_at_k(model, df, failure_modes, actions, k=5)

    logger.info("\\n--- Two-Tower Recommendations: bearing_wear ---")
    recs = recommend_with_two_tower(model, "bearing_wear", failure_modes, actions)
    for r in recs:
        logger.info(f"  {r['action']} (score: {r['score']})")

    torch.save(model.state_dict(), "data/recommendation/two_tower_model.pt")

    logger.info("=" * 60)
    logger.info(f"Two-tower training complete - Precision@5: {precision:.3f}")
    logger.info("=" * 60)

    return model, precision


if __name__ == "__main__":
    run_two_tower_training()
