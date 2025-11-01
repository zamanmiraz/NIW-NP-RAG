from langchain.chat_models import init_chat_model
from niw_np_rag.config.config import GOOGLE_API_KEY
from niw_np_rag.app.rag import RAGPipeline
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from transformers import pipeline
import torch

class LLMRAG:
    def __init__(self, model_name="gemini-2.5-flash", temperature=0.7):
        rag = RAGPipeline(pdfs_path="../../data/uscis_aao_pdfs", vector_store_path="./data/chunks_vector_store", semantic_chunking=True)
        self.retriever = rag.get_retriever(k=5)
        self.model_name = model_name
        self.temperature = temperature
        self.device = 0 if torch.cuda.is_available() else -1
        self.chat_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", google_api_key=GOOGLE_API_KEY)


    def retrieve_context(self, query, k=5):
        docs = self.retriever.invoke(query)
        context = [doc.page_content for doc in docs]
        urls = list({doc.metadata.get("source") for doc in docs if doc.metadata.get("source")})
        return context, urls
    
    def generate_response(self, query):
        context = self.retrieve_context(query, k=5)
        system_prompt = system_prompt = (
    "You are an expert immigration Q&A assistant specializing in National Interest Waiver (NIW) petitions. "
    "Your role is to describe what occurred in the retrieved context, focusing only on the information provided. "
    "You must not add external knowledge, interpretation, or personal evaluation.\n\n"

    "Your task is to summarize what the context shows — such as outcomes, reasoning, or findings — "
    "that relate to NIW petitions and their approval or denial patterns. "
    "Do not make predictions, judgments, or offer advice. "
    "Base your response strictly and exclusively on the retrieved context.\n\n"

    "Always specify your sources clearly by mentioning the document title, case identifier, or link "
    "each time you reference contextual evidence.\n\n"

    "Structure your description according to the three NIW prongs, using only details that appear in the context:\n"
    "1️⃣ **Substantial Merit and National Importance** — Describe how the applicant’s field or work was treated "
    "in the context (e.g., what was considered nationally important or lacking merit). Include examples of outcomes if mentioned.\n"
    "2️⃣ **Well-Positioned to Advance the Proposed Endeavor** — Describe what the context reveals about how petitioners "
    "demonstrated their qualifications, achievements, or future plans, and how USCIS or AAO evaluated those aspects.\n"
    "3️⃣ **Beneficial to the United States** — Describe what the retrieved materials say about how granting the waiver "
    "benefits the U.S., or what reasons were given when such benefit was found insufficient.\n\n"

    "⚠️ Important: You are only describing and summarizing what the retrieved context states. "
    "Do not analyze, speculate, or generate new conclusions beyond it. "
    "If a detail is missing, explicitly say that the context does not contain that information.\n\n"

    "--- Retrieved Context ---\n{context}"
)
        prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate.from_template(
                "Using the following context:\n{context}\nAnswer the question:\n{query}"
            )
        ])
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}"),
        ])
        formatted_prompt = prompt.format_messages(context=context, query=query)
        response = self.chat_model.invoke(formatted_prompt)
        return response