from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from niw_np_rag.config.config import GOOGLE_API_KEY
from niw_np_rag.app.rag import RAGPipeline
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from transformers import pipeline
import torch
import logging
import os
from datetime import datetime

# --- Logging Configuration ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "../../logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "queries.log")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def make_retrieve_context_tool(retriever):
    @tool(response_format="content_and_artifact", description="Retrieve relevant context and URLs from the vector store.")
    def retrieve_context(query: str):
        """Retrieve the most relevant context and URLs from the vector store for the given query."""
        docs = retriever.invoke(query)
        context = [doc.page_content for doc in docs]
        urls = list({doc.metadata.get("source") for doc in docs if doc.metadata.get("source")})
        return context, urls
    return retrieve_context


class LLMRAG:
    def __init__(self, model_name="gemini-2.5-flash", k=100, temperature=0.7):
        rag = RAGPipeline(pdfs_path="../data/uscis_aao_pdfs", vector_store_path="./data/chunks_vector_store", semantic_chunking=True)
        self.retriever = rag.get_retriever(k=k)
        self.retrieve_context = make_retrieve_context_tool(self.retriever)
        self.model_name = model_name
        self.temperature = temperature
        self.device = 0 if torch.cuda.is_available() else -1
        self.chat_model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", google_api_key=GOOGLE_API_KEY)

    # def retrieve_context(self, query):
    #     """Retrieve the most relevant context and URLs from the vector store for the given query."""
    #     docs = self.retriever.invoke(query)
    #     context = [doc.page_content for doc in docs]
    #     urls = list({doc.metadata.get("source") for doc in docs if doc.metadata.get("source")})
    #     return context, urls
    
    def generate_response_evaluator(self, query):
        # context, urls = self.retrieve_context(query)
        system_prompt = system_prompt = (
    "You are an expert immigration Q&A assistant specializing in EB2 National Interest Waiver (NIW) petitions. "
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
    "Always specify your sources clearly by mentioning the document title, case identifier, or link "
    "each time you reference contextual evidence.\n\n"
    "Note: All retrieved context segments share the same URLs, as they originate from the same petitioner’s case.\n\n"
    "--- Retrieved Context ---\n{context} and the source URLs are: {urls}"   
)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}"),
        ])

        # formatted_prompt = prompt.format_messages(context=context, query=query, urls=urls)  # ✅ include URLs
        # response = self.chat_model.invoke(formatted_prompt)

        # ✅ Log query, response, and sources
        # logging.info(
        #     f"QUERY: {query}\nRESPONSE: {response.content[:100]}...\nSOURCES: {', '.join(urls) if urls else 'No sources found'}"
        # )

        # ✅ include both model response and URLs in output
        # return {
        #     "query": query,
        #     "response": response.content if hasattr(response, "content") else str(response),
        #     "sources": urls
        # }
        tools = [self.retrieve_context]
        agent = create_agent(model = self.chat_model, tools=tools, system_prompt=system_prompt, middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
                ),
            ],
        )
        # Stream and capture the final message
        final_output = ""
        for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            message = event["messages"][-1].content
            print(message)  # optional: log in console
            final_output = message

        # return {
        #     "query": query,
        #     "response": final_output,
        #     # "sources": urls
        # }
        return final_output

    def generate_response(self, query):
        # context, urls = self.retrieve_context(query)  # ✅ unpack both

        system_prompt = (
    "You are an expert legal assistant specializing in EB2 National Interest Waiver (NIW) petitions. "
    "Your task is to summarize and explain the information found in the retrieved context, "
    "strictly based on the provided text without adding any external knowledge, opinions, or interpretations.\n\n"
    "When referring to evidence, always cite your sources clearly by including the document title, "
    "case identifier, or source URL each time you reference contextual information.\n\n"
    "Note: All retrieved context segments share the same URLs, as they originate from the same petitioner’s case.\n\n"
    "--- Retrieved Context ---\n{context}\n\n"
    "--- Source URLs ---\n{urls}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{query}"),
        ])

        # formatted_prompt = prompt.format_messages(context=context, query=query, urls=urls)  # ✅ include URLs
        # response = self.chat_model.invoke(formatted_prompt)

        # ✅ Log query, response, and sources
        # logging.info(
        #     f"QUERY: {query}\nRESPONSE: {response.content[:100]}...\nSOURCES: {', '.join(urls) if urls else 'No sources found'}"
        # )

        # ✅ include both model response and URLs in output
        # return {
        #     "query": query,
        #     "response": response.content if hasattr(response, "content") else str(response),
        #     "sources": urls
        # }
        tools = [self.retrieve_context]
        agent = create_agent(model = self.chat_model, tools=tools, system_prompt=system_prompt, middleware=[
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy="redact",
            apply_to_input=True,
        ),
        # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy="mask",
            apply_to_input=True,
        ),
        # Block API keys - raise error if detected
        PIIMiddleware(
            "api_key",
            detector=r"sk-[a-zA-Z0-9]{32}",
            strategy="block",
            apply_to_input=True,
                ),
            ],
        )
        # Stream and capture the final message
        final_output = ""
        for event in agent.stream(
            {"messages": [{"role": "user", "content": query}]},
            stream_mode="values",
        ):
            message = event["messages"][-1].content
            print(message)  # optional: log in console
            final_output = message

        # return {
        #     "query": query,
        #     "response": final_output,
        #     # "sources": urls
        # }
        return final_output
