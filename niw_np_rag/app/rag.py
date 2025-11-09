# app/rag.py
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
# vector stores
from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

from transformers import pipeline
import torch
import logging
import glob
from tqdm import tqdm
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

class RAGPipeline:
    def __init__(self, pdfs_path, vector_store_path = "./data/chunks_vector_store", semantic_chunking=True):
        self.pdfs_path = pdfs_path
        self.vector_store_path = vector_store_path
        # Check and print the absolute path
        abs_path = os.path.abspath(self.vector_store_path)
        self.semantic_chunking = semantic_chunking
        self.device = 0 if torch.cuda.is_available() else -1
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        # self.embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.semantic_chunker = SemanticChunker(self.embedding_model, breakpoint_threshold_type='percentile', breakpoint_threshold_amount=90,)

    # -------------------------------------------------------------------------
    # Filter the pdfs based on date in filename
    # -------------------------------------------------------------------------
    def filter_pdfs_by_date(self, pdf_files, cutoff_date=datetime.datetime(2016, 12, 27)):
        filtered_pdfs = []
        for pdf_file in pdf_files:
            try:
                date_str = os.path.basename(pdf_file).split('_')[0]
                file_date = datetime.datetime.strptime(date_str, '%b%d%Y')
                if file_date > cutoff_date:
                    filtered_pdfs.append(pdf_file)
            except (ValueError, IndexError):
                logging.warning(f"[SKIP] Unexpected filename format: {pdf_file}")
        return filtered_pdfs

    # -------------------------------------------------------------------------
    # Chunk documents with semantic or recursive splitting
    # -------------------------------------------------------------------------
    def chunk_documents(self, pdf, chunk_size=1000, chunk_overlap=100):
        document = PyPDFLoader(pdf).load()
        if self.semantic_chunking:
            texts = self.semantic_chunker.split_documents(document)
        else:
            texts = self.text_splitter.split_documents(document)
        texts = self.text_splitter.split_documents(document)
        return texts
    
    # -------------------------------------------------------------------------
    # Build and update FAISS vector store with progress and filtering
    # -------------------------------------------------------------------------
    def build_vector_store_FAISS(self):
        filtered_pdf_files = self.filter_pdfs_by_date(glob.glob(os.path.join(self.pdfs_path, "*.pdf")))
        # Load existing FAISS store if available
        vector_store = None
        if os.path.exists(self.vector_store_path):
            try:
                logging.info(f"[LOAD] Loading existing FAISS store from {self.vector_store_path}")
                vector_store = FAISS.load_local(
                    self.vector_store_path, self.embedding_model, allow_dangerous_deserialization=True
                )
            except Exception as e:
                logging.error(f"[LOAD ERROR] Failed to load FAISS store: {e}")

        # Process PDFs with progress bar
        for i, pdf_file in enumerate(tqdm(filtered_pdf_files, desc="Processing PDFs")):
            try:
                texts = self.chunk_documents(pdf_file)

                if vector_store is None:
                    vector_store = FAISS.from_documents(texts, self.embedding_model)
                    logging.info(f"[CREATE] New FAISS store created with first file: {pdf_file}")
                else:
                    vector_store.add_documents(texts)
                    logging.info(f"[ADD] Added chunks from: {pdf_file}")

                # Save periodically
                if (i + 1) % 50 == 0:
                    vector_store.save_local(self.vector_store_path)
                    logging.info(f"[SAVE] Saved FAISS after {i + 1} files.")

            except Exception as e:
                logging.error(f"[ERROR] Failed processing {pdf_file}: {e}")

        # Final save
        if vector_store is not None:
            vector_store.save_local(self.vector_store_path)
            logging.info("[COMPLETE] Vector store successfully saved.")
        else:
            logging.warning("[COMPLETE] No vector store was created — check data folder.")

        return vector_store
    
    # -------------------------------------------------------------------------
    # Build vector store using Qdrant
    # -------------------------------------------------------------------------
    def build_vector_store_Qdrant(self):
        client = QdrantClient(path="./data/qdrant_db")  # persistent local db
        collection_name = "niw_chunks"

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1024,  
                distance=Distance.COSINE
            )
        )

        # ✅ Initialize QdrantVectorStore once
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embedding_model,
        )

        filtered_pdf_files = self.filter_pdfs_by_date(glob.glob(os.path.join(self.pdfs_path, "*.pdf")))
        count = 0
        for pdf_file in tqdm(filtered_pdf_files, desc="Processing PDFs for Qdrant"):
            try:
                texts = self.chunk_documents(pdf_file)
                vector_store.add_documents(texts)  # ✅ add to existing collection
                tqdm.write(f"[ADD] Added chunks from: {pdf_file}")
                count += 1
            except Exception as e:
                logging.error(f"[ERROR] Failed processing {pdf_file}: {e}")
        # Final save the vector store
        logging.info("[COMPLETE] All documents added to Qdrant collection.")
        return vector_store

    # -------------------------------------------------------------------------
    # Load vector store and get retriever
    # -------------------------------------------------------------------------
    def load_vector_store(self):
        abs_path = os.path.abspath(self.vector_store_path)
        logging.info(f"Loading vector store from {abs_path}...")
        vector_store = FAISS.load_local(self.vector_store_path, self.embedding_model, allow_dangerous_deserialization=True)
        logging.info(f"Vector store loaded successfully with {vector_store.index.ntotal} vectors.")        
        return vector_store
    
    def get_retriever(self, k):
        vector_store = self.load_vector_store()
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        return retriever