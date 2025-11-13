from niw_np_rag.app.llm_rag import LLMRAG
from niw_np_rag.app.rag import RAGPipeline
from langchain_community.vectorstores import FAISS
import os

def test_llm_rag_initialization():
    llm_rag = LLMRAG()
    print("LLMRAG initialized successfully.")
    response = llm_rag.generate_response("Could you please tell me about the in which different fields the petitioner have been working")
    print("LLMRAG initialized and response generated successfully.")
    print("Response:", response)

def test_rag():
    rag = RAGPipeline(pdfs_path="../data/uscis_aao_pdfs", vector_store_path="../data/chunks_vector_store_faiss", semantic_chunking=True)
    # check if the the pdfs_path is set correctly
    # find the exact path of the pdfs
    ppath = rag.pdfs_path
    # convert to absolute path
    abs_ppath = os.path.abspath(ppath)
    print("RAGPipeline initialized successfully.")
    print(abs_ppath)
    # rag.build_vector_store_Qdrant()
    # print("RAGPipeline vector store built successfully.")
    faiss_store = rag.load_vector_store()
    unique_sources = set()

    for doc in faiss_store.docstore._dict.values():
        if "source" in doc.metadata:
            unique_sources.add(doc.metadata["source"])

    print("Unique PDF files in FAISS:", len(unique_sources))

    # print("Total vectors in FAISS index:", faiss_store.index.ntotal)

if __name__ == "__main__":
    # test_llm_rag_initialization()
    test_rag()