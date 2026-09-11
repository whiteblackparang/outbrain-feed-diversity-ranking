import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

MAX_CANDIDATES = 12


class SessionDataset(Dataset):
    def __init__(self, sessions_df, get_content_vector, embed_dim):
        self.doc_lists = sessions_df["document_id_ad"].tolist()
        self.click_lists = sessions_df["clicked"].tolist()
        self.get_content_vector = get_content_vector
        self.embed_dim = embed_dim

    def __len__(self):
        return len(self.doc_lists)

    def __getitem__(self, idx):
        docs = self.doc_lists[idx][:MAX_CANDIDATES]
        clicks = self.click_lists[idx][:MAX_CANDIDATES]

        vectors = np.stack([self.get_content_vector(d) for d in docs])
        clicked_idx = clicks.index(1)

        return (
            torch.tensor(vectors, dtype=torch.float32),
            torch.tensor(clicked_idx, dtype=torch.long),
            len(docs),
        )


def collate_sessions(batch):
    vectors, clicked_idx, lengths = zip(*batch)
    lengths = torch.tensor(lengths)
    max_len = lengths.max().item()

    padded = torch.zeros(len(batch), max_len, vectors[0].shape[1])
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, v in enumerate(vectors):
        padded[i, :v.shape[0]] = v
        mask[i, :v.shape[0]] = True

    clicked_idx = torch.stack(clicked_idx)
    return padded, mask, clicked_idx


class SessionAttentionModel(nn.Module):
    def __init__(self, input_dim, embed_dim=32, n_heads=4):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, embed_dim))
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, padded, mask):
        x = self.encode(padded)
        key_padding_mask = ~mask
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        scores = self.score(attn_out).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        return scores