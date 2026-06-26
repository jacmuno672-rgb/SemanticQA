import os
import json
import numpy as np

from core.config import KB_PATH

class KnowledgeBase:
    def __init__(self, kb_path=None):
        self.kb_path = kb_path or KB_PATH
        self.entries = []
        self.question_vectors = None
        self._loaded = False

    def build(self, qa_pairs, vectorizer):
        self.entries = []
        questions = []
        for item in qa_pairs:
            if not item.get("question") or not item.get("answer"):
                continue
            questions.append(item["question"])
            self.entries.append(item)

        q_vectors = vectorizer.encode_batch(questions)
        for i, entry in enumerate(self.entries):
            entry["question_vector"] = q_vectors[i].tolist()

        self.question_vectors = q_vectors
        self._loaded = True

        os.makedirs(os.path.dirname(self.kb_path), exist_ok=True)
        with open(self.kb_path, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

        print(f"知识库已构建: {len(self.entries)} 条 QA, 保存至 {self.kb_path}")

    def load(self):
        if not os.path.exists(self.kb_path):
            return False
        with open(self.kb_path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)
        self.question_vectors = np.array([e["question_vector"] for e in self.entries], dtype=np.float32)
        self._loaded = True
        print(f"知识库已加载: {len(self.entries)} 条 QA")
        return True

    def get_vectors(self):
        return self.question_vectors

    def get_entry(self, idx):
        if 0 <= idx < len(self.entries):
            return self.entries[idx]
        return None

    def __len__(self):
        return len(self.entries)
