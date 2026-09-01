"""Runnable hybrid RAG example: chunk, retrieve, fuse, rerank, and answer."""

import re
from collections import Counter
from math import sqrt


# Block 1: Input documents and user question
documents = [
    {
        "source": "HR-LEAVE-847",
        "page": 1,
        "text": (
            "Employees may carry over up to 5 unused PTO days into the next "
            "calendar year. Extra unused days expire on December 31."
        ),
    },
    {
        "source": "HR-LEAVE-210",
        "page": 3,
        "text": (
            "Employees receive 20 paid vacation days each year. New employees "
            "receive a prorated allowance based on their start date."
        ),
    },
    {
        "source": "HR-SICK-105",
        "page": 2,
        "text": (
            "Unused sick leave is separate from PTO and cannot be transferred "
            "to the next calendar year."
        ),
    },
]

query = "Under HR-LEAVE-847, how much unused PTO can I transfer to next year?"


# Block 2: Split documents into smaller searchable chunks
def chunk_text(text, chunk_size=30):
    words = text.split()
    return [
        " ".join(words[start:start + chunk_size])
        for start in range(0, len(words), chunk_size)
    ]


def create_chunks(input_documents):
    chunks = []
    for document in input_documents:
        for number, piece in enumerate(chunk_text(document["text"]), start=1):
            chunks.append({
                "id": f"{document['source']}#chunk-{number}",
                "source": document["source"],
                "page": document["page"],
                "text": piece,
            })
    return chunks


# Block 3: Clean and expand the query
def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def process_query(user_query):
    replacements = {
        "pto": "paid time off annual leave vacation",
        "transfer": "carry over rollover",
        "next year": "next calendar year",
    }
    semantic_query = user_query.lower()
    for original, expansion in replacements.items():
        semantic_query = semantic_query.replace(original, expansion)
    return {
        "original": user_query,
        "semantic_query": semantic_query,
        "keywords": tokenize(user_query),
    }


# Block 4: Dense-style vector search with character features
def character_features(text):
    cleaned = " ".join(tokenize(text))
    return Counter(cleaned[i:i + 3] for i in range(max(0, len(cleaned) - 2)))


def vectorize(feature_counts, vocabulary):
    return [float(feature_counts.get(feature, 0)) for feature in vocabulary]


def normalize(vector):
    length = sqrt(sum(value * value for value in vector))
    return [value / length for value in vector] if length else vector


def cosine_similarity(first, second):
    return sum(
        left * right
        for left, right in zip(normalize(first), normalize(second))
    )


def dense_search(search_query, chunks):
    query_features = character_features(search_query)
    chunk_features = [character_features(chunk["text"]) for chunk in chunks]
    vocabulary = sorted(
        set(query_features).union(*(set(features) for features in chunk_features))
    )
    query_vector = vectorize(query_features, vocabulary)
    scores = [
        cosine_similarity(vectorize(features, vocabulary), query_vector)
        for features in chunk_features
    ]
    ranking = sorted(range(len(chunks)), key=scores.__getitem__, reverse=True)
    return ranking, scores


# Block 5: Sparse search with exact word overlap
def sparse_search(keywords, chunks):
    query_words = Counter(keywords)
    scores = [
        sum((Counter(tokenize(chunk["text"])) & query_words).values())
        for chunk in chunks
    ]
    ranking = sorted(range(len(chunks)), key=scores.__getitem__, reverse=True)
    return ranking, scores


# Block 6: Combine rankings with Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(dense_ranking, sparse_ranking, k=60):
    scores = {}
    for ranking in (dense_ranking, sparse_ranking):
        for rank, chunk_index in enumerate(ranking, start=1):
            old_score = scores.get(chunk_index, 0)
            scores[chunk_index] = old_score + 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


# Block 7: Rerank, rewarding word overlap and an exact source-ID match
def rerank(user_query, fused_results, chunks):
    query_tokens = set(tokenize(user_query))
    reranked = []
    for chunk_index, rrf_score in fused_results:
        chunk = chunks[chunk_index]
        overlap = len(query_tokens & set(tokenize(chunk["text"])))
        source_match = chunk["source"].lower() in user_query.lower()
        score = rrf_score + overlap * 0.01 + int(source_match) * 0.1
        reranked.append((chunk_index, score))
    return sorted(reranked, key=lambda item: item[1], reverse=True)


# Block 8: Build context and create a grounded extractive answer
def build_context(selected_chunks):
    blocks = []
    for number, chunk in enumerate(selected_chunks, start=1):
        blocks.append(
            f"[Source {number}: {chunk['source']}, page {chunk['page']}]\n"
            f"{chunk['text']}"
        )
    return "\n\n".join(blocks)


def generate_grounded_answer(selected_chunks):
    """A real RAG application would call an LLM at this point."""
    if not selected_chunks:
        return "Not enough information."
    best_chunk = selected_chunks[0]
    amount = re.search(r"up to \d+ unused PTO days", best_chunk["text"], re.I)
    if amount:
        return f"You can carry over {amount.group(0)} [Source 1]."
    return "Not enough information."


# Block 9: Run every retrieval stage in the correct order
def answer_question(user_query, input_documents, top_k=2):
    chunks = create_chunks(input_documents)
    processed = process_query(user_query)
    dense_ranking, dense_scores = dense_search(processed["semantic_query"], chunks)
    sparse_ranking, sparse_scores = sparse_search(processed["keywords"], chunks)
    fused = reciprocal_rank_fusion(dense_ranking, sparse_ranking)
    ranked = rerank(processed["original"], fused, chunks)
    selected_chunks = [chunks[index] for index, _ in ranked[:top_k]]

    return {
        "answer": generate_grounded_answer(selected_chunks),
        "context": build_context(selected_chunks),
        "ranked_results": [
            {
                "chunk": chunks[index],
                "rerank_score": rerank_score,
                "dense_score": dense_scores[index],
                "sparse_score": sparse_scores[index],
            }
            for index, rerank_score in ranked
        ],
    }


# Block 10: Print results when this file is run directly
if __name__ == "__main__":
    result = answer_question(query, documents)
    print(f"Question: {query}")
    print(f"Answer: {result['answer']}\n")
    print("Retrieved context:")
    print(result["context"])
    print("\nRanking details:")
    for position, item in enumerate(result["ranked_results"], start=1):
        print(
            f"{position}. {item['chunk']['id']} | "
            f"rerank={item['rerank_score']:.6f} | "
            f"dense={item['dense_score']:.6f} | "
            f"sparse={item['sparse_score']}"
        )
