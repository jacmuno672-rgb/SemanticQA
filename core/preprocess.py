import os
import json
import jieba

from core.config import RAW_DATA_DIR

def load_raw_texts(data_dir=None):
    if data_dir is None:
        data_dir = RAW_DATA_DIR

    texts = []
    if not os.path.exists(data_dir):
        return texts

    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if not fname.endswith(".txt") and not fname.endswith(".json"):
            continue
        if fname.endswith(".json"):
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        texts.append(item.get("question", ""))
                        texts.append(item.get("answer", ""))
                elif isinstance(data, dict):
                    for v in data.values():
                        if isinstance(v, list):
                            for item in v:
                                if isinstance(item, dict):
                                    texts.append(item.get("question", ""))
                                    texts.append(item.get("answer", ""))
        else:
            with open(fpath, "r", encoding="utf-8") as f:
                texts.extend([line.strip() for line in f if line.strip()])

    return texts

def tokenize(text):
    return list(jieba.cut(text))

def tokenize_all(texts):
    return [tokenize(t) for t in texts]

def build_vocab(tokenized_texts, min_count=2):
    word_counts = {}
    for tokens in tokenized_texts:
        for w in tokens:
            word_counts[w] = word_counts.get(w, 0) + 1

    words = ["<PAD>", "<UNK>"]
    for w, c in sorted(word_counts.items(), key=lambda x: -x[1]):
        if c >= min_count:
            words.append(w)

    word2idx = {w: i for i, w in enumerate(words)}
    return word2idx, words
