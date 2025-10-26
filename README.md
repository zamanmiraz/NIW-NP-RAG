# NIW-NP-RAG

## Retrieval-Augmented Generation (RAG) Pipeline for Analyzing USCIS AAO PDF Case Documents

This repository contains a Google Colab notebook demonstrating a Retrieval-Augmented Generation (RAG) pipeline designed to process and query USCIS Administrative Appeals Office (AAO) PDF case documents. The pipeline leverages semantic chunking and vector embeddings to enable intelligent question answering based on the content of the documents.

## Features

- **PDF Document Loading:** Load multiple PDF files from a specified directory.
- **Text Preprocessing:** Clean the extracted text content.
- **Semantic Chunking:** Utilize `SemanticChunker` with Hugging Face embeddings for intelligent text splitting based on semantic similarity.
- **Vector Store Creation:** Build a FAISS vector database to store and index the document embeddings.
- **Context Retrieval:** Efficiently retrieve relevant document chunks based on a user query.
- **RAG Question Answering:** Combine a language model (Gemini 2.5 Flash) with the retrieved context to answer user questions about the documents.
