# app/rag.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import FAISS
from transformers import pipeline
import torch

class RAGPipeline:
    def __init__(self, pdf_path, vector_store_path = "data/faiss_store", semantic_chunking=True):
        self.pdf_path = pdf_path
        self.vector_store_path = vector_store_path
        self.semantic_chunking = semantic_chunking
        self.device = 0 if torch.cuda.is_available() else -1
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.semantic_chunker = SemanticChunker(model_name=self.embedding_model, chunk_size=500, overlap_size=50, embedding_function=self.embedding_model)

    def chunk_documents_splitter(self, pdf, chunk_size=1000, chunk_overlap=100):
        document = PyPDFLoader(pdf).load()
        texts = self.text_splitter.split_documents(document)
        return texts
    
    def chunk_documents_semantic(self, pdf):
        document = PyPDFLoader(pdf).load()
        texts = self.semantic_chunker.split_documents(document)
        return texts
