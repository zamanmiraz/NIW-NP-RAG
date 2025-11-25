import numpy as np
from niw_np_rag.app.rag import RAGPipeline
import json

with open(r".\evaluation\datasets\niw_qna.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

# Initialize your RAG pipeline
rag = RAGPipeline(
    pdfs_path="./data/uscis_aao_pdfs",
    vector_store_path="./data/chunks_vector_store_faiss",
    semantic_chunking=True
)

def distance(a, list_b):
    """Compute minimum Levenshtein distance between string a and any string in list_b."""
    from Levenshtein import distance as lev_distance
    return min(lev_distance(a, b) for b in list_b)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

retriever = rag.get_retriever(k=5)


def evaluate_recall_at_k(dataset, retriever, k=5):
    """
    dataset: list of dicts with fields:
        - question
        - answer  (not required for recall)
        - context (ground-truth context from source docs)
    """

    hits = []

    for item in dataset:
        question = item["question"]
        ground_truth_context = item["context"]
        # print(question, ground_truth_context)

        # Retrieve top-k documents
        retrieved_docs = retriever.invoke(question)

        # Extract retrieved text
        retrieved_texts = [doc.page_content for doc in retrieved_docs[:k]]

        # Check if ground-truth context appears in retrieved docs
        hit = any(
            distance (ground_truth_context, retrieved_texts) in retrieved_doc
            for retrieved_doc in retrieved_texts
        )

        hits.append(1 if hit else 0)

    recall_k = np.mean(hits)

    print(f"Recall@{k}: {recall_k:.4f}")
    return recall_k

evaluate_recall_at_k(dataset, retriever, k=15)