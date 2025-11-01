from fastapi import FastAPI
from niw_np_rag.app.llm_rag import LLMRAG
app = FastAPI(title="NIW-NP-RAG", version="0.1.0")

@app.get("/health")

def health_check():
    return {"status": "ok"}

@app.get("/query")
def ask_question(query: str):
    llm_rag = LLMRAG()
    response = llm_rag.generate_response(query)
    return {"query": query, "response": response.content}