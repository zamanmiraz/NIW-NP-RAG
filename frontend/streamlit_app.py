import streamlit as st
import requests

FASTAPI_URL = "http://localhost:8000"  # FastAPI backend

st.set_page_config(page_title="NIW-NP-RAG Explorer", page_icon="📄", layout="wide")

st.title("📘 NIW-NP-RAG — Interactive Legal Document Explorer")

st.markdown("""
Use the app to either **ask questions** about USCIS AAO NIW Non-Precedent decisions  
or **evaluate** your case against past rulings — fully offline.
""")

# --- User Mode Selection ---
st.markdown("### ⚙️ Choose what you want to do:")
mode = st.radio(
    "Select mode:",
    ["Ask a Question", "Evaluate a Case"],
    horizontal=True,
    index=0
)

# # --- Query Input ---
# query = st.text_input(
#     "Enter your question or case description:",
#     placeholder="e.g., What are the main reasons for denials?"
# )

# submit = st.button("🚀 Run")

# # --- Backend Request ---
# if submit and query:
#     with st.spinner("Processing your request..."):
#         try:
#             if mode == "Ask a Question":
#                 endpoint = f"{FASTAPI_URL}/query"
#             else:
#                 endpoint = f"{FASTAPI_URL}/evaluate"

#             response = requests.get(endpoint, params={"query": query})
            
#             if response.status_code == 200:
#                 result = response.json()
#                 st.subheader("🧠 Response:")
#                 st.write(result["response"])

#                 with st.expander("📜 Full Response JSON"):
#                     st.json(result)
#             else:
#                 st.error(f"❌ Backend returned {response.status_code}: {response.text}")
#         except Exception as e:
#             st.error(f"⚠️ Error connecting to backend: {e}")

# st.markdown("---")
st.caption("NIW-NP-RAG | Local RAG-powered legal reasoning system")
