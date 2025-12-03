import numpy as np
from niw_np_rag.app.rag import RAGPipeline
import json
from langchain_huggingface import HuggingFaceEmbeddings
import math

with open(r".\evaluation\datasets\niw_qna.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Initialize your RAG pipeline
rag = RAGPipeline(
    pdfs_path="./data/uscis_aao_pdfs",
    vector_store_path="./data/chunks_vector_store_faiss",
    semantic_chunking=True
)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

retriever = rag.get_retriever(k=5)

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# -----------------------------
# Utility Functions
# -----------------------------
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def is_hit(gt_context, retrieved_doc, threshold=0.45):
    """Semantic match check using embeddings."""
    gt_emb = emb.embed_query(gt_context)
    doc_emb = emb.embed_query(retrieved_doc)
    sim = cosine_similarity(gt_emb, doc_emb)
    return sim >= threshold


# -----------------------------
# Retrieval Metrics
# -----------------------------
def evaluate_retrieval(dataset, retriever, k=10):
    hits = []               # for Recall@K
    precision_scores = []   # for Precision@K
    mrr_scores = []         # for Mean Reciprocal Rank
    ndcg_scores = []        # for nDCG@K

    for item in dataset:
        question = item["question"]
        ground_truth_context = item["context"]

        # Retrieve documents (we keep top-K)
        retrieved_docs = retriever.invoke(question)
        retrieved_texts = [doc.page_content for doc in retrieved_docs[:k]]

        # Track hits at ranks
        hit_list = []

        for idx, doc_text in enumerate(retrieved_texts):
            match = is_hit(ground_truth_context, doc_text)

            hit_list.append(1 if match else 0)

        # -----------------------------
        # Compute Metrics
        # -----------------------------

        # Recall@K → was any retrieved doc correct?
        recall_k = 1 if any(hit_list) else 0
        hits.append(recall_k)

        # Precision@K → proportion of correct retrieved docs
        precision = sum(hit_list) / k
        precision_scores.append(precision)

        # MRR → 1 / rank of first relevant doc
        if 1 in hit_list:
            rank = hit_list.index(1) + 1
            mrr_scores.append(1 / rank)
        else:
            mrr_scores.append(0)

        # nDCG@K → ranking quality
        dcg = sum([
            hit_list[i] / math.log2(i + 2)  # DCG formula
            for i in range(len(hit_list))
        ])

        # Ideal DCG (all relevant docs ranked at top)
        ideal_hits = sorted(hit_list, reverse=True)
        idcg = sum([
            ideal_hits[i] / math.log2(i + 2)
            for i in range(len(ideal_hits))
        ])

        ndcg = dcg / idcg if idcg > 0 else 0
        ndcg_scores.append(ndcg)

    # -----------------------------
    # Final Averages
    # -----------------------------
    results = {
        "Recall@K": float(np.mean(hits)),
        "Precision@K": float(np.mean(precision_scores)),
        "MRR": float(np.mean(mrr_scores)),
        "nDCG@K": float(np.mean(ndcg_scores)),
    }

    return results


# -----------------------------
# Run Evaluation
# -----------------------------
metrics = evaluate_retrieval(dataset, retriever, k=15)

print("\n=== Retrieval Evaluation Results ===")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")