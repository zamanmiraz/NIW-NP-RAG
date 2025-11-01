# app/rag.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline
import torch

class RAGPipeline:
    def __init__(self, pdfs_path, vector_store_path = "./data/chunks_vector_store", semantic_chunking=True):
        self.pdfs_path = pdfs_path
        self.vector_store_path = vector_store_path
        # ✅ Check and print the absolute path
        abs_path = os.path.abspath(self.vector_store_path)
        print(f"[INFO] Vector store path: {abs_path}")
        self.semantic_chunking = semantic_chunking
        self.device = 0 if torch.cuda.is_available() else -1
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.semantic_chunker = SemanticChunker(self.embedding_model, breakpoint_threshold_type='percentile', breakpoint_threshold_amount=90,)

    def chunk_documents(self, pdf, chunk_size=1000, chunk_overlap=100):
        document = PyPDFLoader(pdf).load()
        if self.semantic_chunking:
            texts = self.semantic_chunker.split_documents(document)
        else:
            texts = self.text_splitter.split_documents(document)
        texts = self.text_splitter.split_documents(document)
        return texts
    
    def build_vector_store(self):
        all_texts = []
        for filename in os.listdir(self.pdfs_path):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(self.pdfs_path, filename)
                texts = self.chunk_documents(pdf_path)
                all_texts.extend(texts)
        vector_store = FAISS.from_documents(all_texts, self.embedding_model)
        vector_store.save_local(self.vector_store_path)
        return vector_store
    
    def load_vector_store(self):
        vector_store = FAISS.load_local(self.vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)
        print("Vector store loaded with", vector_store.index.ntotal, "vectors.")
        return vector_store
    
    def get_retriever(self, k=5):
        vector_store = self.load_vector_store()
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        return retriever