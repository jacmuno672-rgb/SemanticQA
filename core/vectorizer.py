import json
import math
import numpy as np

from core.config import W2V_MODEL_PATH
from core.preprocess import tokenize

class Vectorizer:
    def __init__(self, w2v_path=None):
        self.w2v_path = w2v_path or W2V_MODEL_PATH
        self.vectors = None
        self.word2idx = None
        self.embed_dim = None
        self.idf = {}

    def load(self):
        import torch
        data = torch.load(self.w2v_path, map_location="cpu", weights_only=False)
        self.vectors = data["vectors"]
        self.word2idx = data["word2idx"]
        self.embed_dim = data["embed_dim"]
        print(f"向量化器已加载: {self.vectors.shape[0]} 词, 维度 {self.embed_dim}")

    def compute_idf(self, texts):
        df = {}
        total = 0
        for text in texts:
            tokens = set(tokenize(text))
            total += 1
            for w in tokens:
                df[w] = df.get(w, 0) + 1
        self.idf = {w: math.log((total + 1) / (c + 1)) + 1.0 for w, c in df.items()}

    def word2vec(self, word):
        if word in self.word2idx:
            idx = self.word2idx[word]
            return self.vectors[idx].numpy() if hasattr(self.vectors, "numpy") else np.array(self.vectors[idx])
        idx = self.word2idx.get("<UNK>", 1)
        if hasattr(self.vectors, "numpy"):
            return self.vectors[idx].numpy()
        return np.array(self.vectors[idx])

    def sentence2vec(self, text):
        tokens = tokenize(text)
        if not tokens:
            return np.zeros(self.embed_dim, dtype=np.float32)
        weights = []
        vecs = []
        for w in tokens:
            vecs.append(self.word2vec(w))
            weights.append(self.idf.get(w, 1.0))
        if not weights or sum(weights) == 0:
            return np.zeros(self.embed_dim, dtype=np.float32)
        vecs = [v * w for v, w in zip(vecs, weights)]
        return np.sum(vecs, axis=0) / sum(weights)

    def encode_batch(self, texts):
        return np.array([self.sentence2vec(t) for t in texts], dtype=np.float32)
