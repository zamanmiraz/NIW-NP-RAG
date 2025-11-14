from fastapi import FastAPI
from niw_np_rag.app.llm_rag import LLMRAG

app = FastAPI(title="NIW-NP-RAG", version="0.1.0")

print("Starting NIW-NP-RAG")
@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def home():
    return {"message": "NIW-NP-RAG API is running 🚀", "endpoints": ["/health", "/query", "/evaluate"]}


# @app.get("/query")
# def ask_question(query: str):
#     llm_rag = LLMRAG(k=5)
#     response = llm_rag.generate_response(query)
#     return {"query": query, "response": response.content}

@app.get("/query")
def ask_question(query: str):
    llm_rag = LLMRAG(k=5)
    response = llm_rag.generate_response(query)
    return {"query": query, "response": response}


@app.get("/evaluate")
def evaluate_question(query: str):
    llm_rag = LLMRAG(k=5)
    response = llm_rag.generate_response_evaluator(query)
    return {"query": query, "response": response}