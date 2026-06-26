import os

from core.config import KB_PATH, W2V_MODEL_PATH, TOP_K
from core.knowledge_base import KnowledgeBase
from core.vectorizer import Vectorizer
from core.similarity import compute_similarity


class QAEngine:
    def __init__(self):
        self.kb = KnowledgeBase(KB_PATH)
        self.vectorizer = Vectorizer(W2V_MODEL_PATH)

    def init(self):
        if os.path.exists(W2V_MODEL_PATH) and os.path.exists(KB_PATH):
            self.vectorizer.load()
            self.kb.load()
            questions = [e["question"] for e in self.kb.entries]
            self.vectorizer.compute_idf(questions)
            return True
        return False

    def ask(self, question, top_k=TOP_K):
        q_vec = self.vectorizer.sentence2vec(question)
        indices, scores = compute_similarity(q_vec, self.kb.get_vectors(), top_k)
        results = []
        for idx, score in zip(indices, scores):
            entry = self.kb.get_entry(idx)
            if entry:
                results.append({
                    "question": entry["question"],
                    "answer": entry["answer"],
                    "category": entry.get("category", ""),
                    "score": round(float(score), 4)
                })
        return results

    def ask_single(self, question):
        results = self.ask(question, top_k=1)
        if results:
            return results[0]
        return {"question": question, "answer": "抱歉，没有找到匹配的答案。", "score": 0}
