import argparse
import json
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util

"""
create_rag_eval_dataset.py

Generates a realistic RAG evaluation dataset for your EB2-NIW legal assistant system.

For each AAO PDF:
  1. Extract and chunk text
  2. Generate legal-style queries (based on your system prompt)
  3. Retrieve top-k chunks as "retrieved context" for each query
  4. Save dataset entries usable for RAG evaluation

Output JSONL structure:
{
  "case_id": "...",
  "query": "...",
  "retrieved_context": "...",
  "urls": ["https://..."],
  "source_file": "...",
  "metadata": {...}
}

Usage:
    python evaluation/create_rag_eval_dataset.py -i ./data/aao_cases -o ./evaluation/datasets/rag_eval.jsonl
"""

# ---------- Text extraction ----------

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise ImportError("Install PyMuPDF: pip install PyMuPDF") from e


def extract_text_from_pdf(path: Path) -> str:
    """Extract text from PDF."""
    doc = fitz.open(str(path))
    text = "\n".join([p.get_text("text") for p in doc])
    doc.close()
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- Chunking ----------

def chunk_text(text: str, max_words: int = 180, overlap: int = 40) -> List[str]:
    """Simple word-based chunker."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


# ---------- Query generation ----------

def generate_queries(text: str, n: int = 3) -> List[str]:
    """
    Generate EB2-NIW specific questions for RAG evaluation.
    """
    base_templates = [
        "Summarize the petitioner's field of work and claimed national interest.",
        "Explain the main reasons the AAO reached its decision and supporting rationale.",
        "List the key evidence cited and how the AAO interpreted that evidence.",
        "Describe the procedural history and the outcome ordered by the AAO.",
        "Identify the legal standards or precedents the AAO applied and how they influenced the result.",
    ]
    return base_templates[:n]


# ---------- Context retrieval ----------

def retrieve_context(queries: List[str], chunks: List[str], model, top_k: int = 3) -> Dict[str, List[str]]:
    """
    Retrieve top-k most relevant chunks for each query using embeddings similarity.
    """
    chunk_embs = model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
    query_to_context = {}

    for query in queries:
        q_emb = model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(q_emb, chunk_embs)[0]
        top_results = cos_scores.topk(k=min(top_k, len(chunks)))
        retrieved = [chunks[i] for i in top_results[1]]
        query_to_context[query] = retrieved

    return query_to_context


# ---------- Main dataset creator ----------

def create_dataset(input_dir: Path, output_file: Path, model_name: str, n_queries: int = 3, top_k: int = 3):
    model = SentenceTransformer(model_name)
    pdf_files = list(input_dir.rglob("*.pdf"))
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            text = extract_text_from_pdf(pdf_path)
        except Exception as e:
            print(f"Failed to extract {pdf_path.name}: {e}")
            continue

        if not text:
            continue

        chunks = chunk_text(text)
        queries = generate_queries(text, n=n_queries)
        q_to_ctx = retrieve_context(queries, chunks, model, top_k=top_k)

        url_stub = f"https://www.uscis.gov/administrative-appeals/aaodecisions/{pdf_path.stem}"

        for query, contexts in q_to_ctx.items():
            record = {
                "case_id": pdf_path.stem,
                "query": query,
                "retrieved_context": "\n\n".join(contexts),
                "urls": [url_stub],
                "source_file": str(pdf_path),
                "metadata": {"case_type": "EB2-NIW"},
            }
            records.append(record)

    with output_file.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n✅ Created {len(records)} query-context pairs at {output_file}")


# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate RAG evaluation dataset with retrieved context")
    parser.add_argument("--input-dir", "-i", type=Path, required=True, help="Directory with AAO PDFs")
    parser.add_argument("--output", "-o", type=Path, default=Path("evaluation/datasets/rag_eval.jsonl"), help="Output JSONL")
    parser.add_argument("--model", "-m", type=str, default="BAAI/bge-small-en-v1.5", help="SentenceTransformer model")
    parser.add_argument("--queries", "-q", type=int, default=3, help="Queries per document")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k context chunks to retrieve per query")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    create_dataset(
        input_dir=args.input_dir,
        output_file=args.output,
        model_name=args.model,
        n_queries=args.queries,
        top_k=args.top_k,
    )
