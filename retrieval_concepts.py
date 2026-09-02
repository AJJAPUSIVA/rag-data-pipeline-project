"""Learn the main retrieval concepts in one runnable file.

Setup and run:
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python retrieval_concepts.py
"""

import math
import re
from collections import Counter

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer


# 1. Documents and metadata
documents = [
    {"text": "Employees may carry five unused vacation days.", "team": "hr"},
    {"text": "Annual leave rollover is limited to five days.", "team": "hr"},
    {"text": "Password resets require email verification.", "team": "it"},
    {"text": "Expense claims must be submitted within 30 days.", "team": "finance"},
    {"text": "Unused holiday allowance above five days expires.", "team": "hr"},
    {"text": "Verify your email before changing a password.", "team": "it"},
]

query = "How many vacation days can carry over?"
texts = [document["text"] for document in documents]


# 2. Embeddings: text -> numerical vectors
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
document_vectors = embedding_model.encode(texts, normalize_embeddings=True)
document_vectors = np.asarray(document_vectors, dtype="float32")
dimension = document_vectors.shape[1]


def embed(text):
    """Normalize so dot product equals cosine similarity."""
    vector = embedding_model.encode(text, normalize_embeddings=True)
    return np.asarray([vector], dtype="float32")


def format_results(scores, ids):
    return [
        {"id": int(doc_id), "score": float(score), "text": texts[doc_id]}
        for score, doc_id in zip(scores[0], ids[0])
        if doc_id >= 0
    ]


# 3. KNN: exact search checks every vector
exact_index = faiss.IndexFlatIP(dimension)
exact_index.add(document_vectors)


def exact_knn_search(question, top_k=3):
    scores, ids = exact_index.search(embed(question), top_k)
    return format_results(scores, ids)


# 4. ANN with HNSW: fast graph traversal
hnsw_index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
hnsw_index.hnsw.efConstruction = 40
hnsw_index.hnsw.efSearch = 16
hnsw_index.add(document_vectors)


def hnsw_search(question, top_k=3):
    scores, ids = hnsw_index.search(embed(question), top_k)
    return format_results(scores, ids)


# 5. ANN with IVF: search only the nearest vector clusters
cluster_index = faiss.IndexFlatIP(dimension)
ivf_index = faiss.IndexIVFFlat(
    cluster_index, dimension, 2, faiss.METRIC_INNER_PRODUCT
)
ivf_index.train(document_vectors)  # IVF must learn its clusters first.
ivf_index.add(document_vectors)
ivf_index.nprobe = 1  # Search one cluster; more clusters improve recall but cost time.


def ivf_search(question, top_k=3):
    scores, ids = ivf_index.search(embed(question), top_k)
    return format_results(scores, ids)


# 6. Metadata filtering: restrict candidates before vector comparison
def metadata_search(question, team, top_k=3):
    allowed_ids = [
        doc_id for doc_id, document in enumerate(documents)
        if document["team"] == team
    ]
    if not allowed_ids:
        return []

    # Matrix multiplication computes every candidate's dot product.
    scores = document_vectors[allowed_ids] @ embed(question)[0]
    order = np.argsort(-scores)[:top_k]
    return [
        {
            "id": allowed_ids[position],
            "score": float(scores[position]),
            "text": texts[allowed_ids[position]],
        }
        for position in order
    ]


# 7. BM25: sparse keyword relevance
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_scores(question):
    tokenized_documents = [tokenize(text) for text in texts]
    query_terms = tokenize(question)
    average_length = sum(map(len, tokenized_documents)) / len(documents)
    document_frequency = Counter(
        term for terms in tokenized_documents for term in set(terms)
    )

    scores = []
    for terms in tokenized_documents:
        frequencies = Counter(terms)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            idf = math.log(
                1 + (len(documents) - document_frequency[term] + 0.5)
                / (document_frequency[term] + 0.5)
            )
            denominator = frequency + 1.5 * (
                0.25 + 0.75 * len(terms) / average_length
            )
            score += idf * frequency * 2.5 / denominator
        scores.append(score)
    return np.asarray(scores, dtype="float32")


# 8. Hybrid retrieval: fuse vector and BM25 rankings
def hybrid_search(question, top_k=4):
    dense_ids = [result["id"] for result in exact_knn_search(question, len(documents))]
    sparse_ids = np.argsort(-bm25_scores(question)).tolist()

    # Reciprocal Rank Fusion combines ranks even though score scales differ.
    fused_scores = Counter()
    for ranking in (dense_ids, sparse_ids):
        for rank, doc_id in enumerate(ranking, start=1):
            fused_scores[doc_id] += 1 / (60 + rank)

    return [
        {"id": doc_id, "score": score, "text": texts[doc_id]}
        for doc_id, score in fused_scores.most_common(top_k)
    ]


# 9. Reranking: expensive second-stage scoring of a small candidate set
def rerank(question, candidates):
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    scores = reranker.predict([(question, item["text"]) for item in candidates])
    results = [
        {**item, "rerank_score": float(score)}
        for item, score in zip(candidates, scores)
    ]
    return sorted(results, key=lambda item: item["rerank_score"], reverse=True)


# 10. Recall@K: fraction of true relevant items recovered in the first K
def recall_at_k(true_ids, returned_ids, k):
    if not true_ids:
        raise ValueError("true_ids cannot be empty")
    return len(set(true_ids) & set(returned_ids[:k])) / len(set(true_ids))


if __name__ == "__main__":
    exact_results = exact_knn_search(query)
    hnsw_results = hnsw_search(query)
    ivf_results = ivf_search(query)
    hybrid_results = hybrid_search(query)

    print("\n1. Exact KNN:", exact_results)
    print("\n2. HNSW ANN:", hnsw_results)
    print("\n3. IVF ANN:", ivf_results)
    print("\n4. Metadata filter (team=hr):", metadata_search(query, "hr"))
    print("\n5. Hybrid vector + BM25:", hybrid_results)
    print("\n6. Reranked hybrid results:", rerank(query, hybrid_results))

    exact_ids = [item["id"] for item in exact_results]
    hnsw_ids = [item["id"] for item in hnsw_results]
    print("\n7. HNSW Recall@3:", recall_at_k(exact_ids, hnsw_ids, 3))
