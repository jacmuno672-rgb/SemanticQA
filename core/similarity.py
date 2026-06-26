import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(query_vector, knowledge_vectors, top_k=5):
    if len(knowledge_vectors) == 0:
        return [], []

    query_vector = query_vector.reshape(1, -1)
    sims = cosine_similarity(query_vector, knowledge_vectors)[0]

    if top_k > len(sims):
        top_k = len(sims)

    top_indices = np.argsort(sims)[::-1][:top_k]
    top_scores = sims[top_indices]

    return top_indices.tolist(), top_scores.tolist()
