import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from core.config import MODEL_DIR, W2V_MODEL_PATH, VOCAB_PATH, EMBED_DIM, WINDOW_SIZE, MIN_COUNT, EPOCHS, BATCH_SIZE, LR

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.out_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, context):
        embeds = self.in_embed(context)
        x = embeds.mean(dim=1)
        out = x @ self.out_embed.weight.t()
        return out

    def get_vectors(self):
        return (self.in_embed.weight.data + self.out_embed.weight.data) / 2


class CBOWDataset(Dataset):
    def __init__(self, tokenized_texts, word2idx, window_size):
        self.data = []
        for tokens in tokenized_texts:
            ids = [word2idx.get(w, word2idx["<UNK>"]) for w in tokens]
            for i, target in enumerate(ids):
                ctx = []
                for j in range(i - window_size, i + window_size + 1):
                    if j == i or j < 0 or j >= len(ids):
                        continue
                    ctx.append(ids[j])
                if ctx:
                    while len(ctx) < window_size * 2:
                        ctx.append(0)
                    ctx = ctx[:window_size * 2]
                    self.data.append((ctx, target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ctx, target = self.data[idx]
        return torch.tensor(ctx, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def train_word2vec(tokenized_texts, word2idx, vocab_size):
    dataset = CBOWDataset(tokenized_texts, word2idx, WINDOW_SIZE)
    if len(dataset) == 0:
        return None

    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True, drop_last=True)

    model = CBOWModel(vocab_size, EMBED_DIM).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    best_loss = float("inf")
    patience_counter = 0

    print(f"训练 Word2Vec (CBOW) - 样本数: {len(dataset)}, 词汇量: {vocab_size}, 维度: {EMBED_DIM}")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for ctx, target in loader:
            ctx, target = ctx.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(ctx)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        if epoch % 40 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}  Loss: {avg_loss:.4f}")

        if avg_loss < best_loss - 1e-5:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 60:
                print(f"  Early stopping at epoch {epoch}")
                break

    vectors = model.get_vectors().cpu()
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({"vectors": vectors, "word2idx": word2idx, "embed_dim": EMBED_DIM}, W2V_MODEL_PATH)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump({"word2idx": word2idx, "embed_dim": EMBED_DIM}, f, ensure_ascii=False, indent=2)

    print(f"Word2Vec 模型已保存: {W2V_MODEL_PATH}")
    return model
