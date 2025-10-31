from langchain.chat_models import init_chat_model
from config.config import GOOGLE_API_KEY
from app.rag import RAGPipeline
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from transformers import pipeline
import torch

class LLMRAG:
    def __init__(self, model_name="gemini-2.5-flash", temperature=0.7):
        rag = RAGPipeline(pdfs_path="../../data/uscis_aao_pdfs", vector_store_path="../../data/chunks_vector_store", semantic_chunking=True)
        self.retriever = rag.get_retriever(k=5)
        self.model_name = model_name
        self.temperature = temperature
        self.device = 0 if torch.cuda.is_available() else -1
        self.chat_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", google_api_key=GOOGLE_API_KEY)


    def retrieve_context(self, query, k=5):
        docs = self.retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        return context
    
    def generate_response(self, query):
        context = self.retrieve_context(query, k=5)
        prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                "Using the following context:\n{context}\nAnswer the question:\n{query}"
            )
        ])
        formatted_prompt = prompt.format_messages(context=context, query=query)
        response = self.chat_model(formatted_prompt)
        return response