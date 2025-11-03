import streamlit as st
import requests

# --- Backend URL ---
FASTAPI_URL = "http://localhost:8000"

# --- Page Config ---
st.set_page_config(page_title="NIW-NP-RAG Explorer", page_icon="📄", layout="wide")

# --- Title and Description ---
st.title("📘 NIW-NP-RAG — Interactive Legal Document Explorer")
st.markdown("""
Use the app to either **ask questions** about USCIS AAO NIW Non-Precedent decisions  
or **evaluate** your case against past rulings — fully offline.
""")

# --- Mode Selection ---
st.markdown("### ⚙️ Choose what you want to do:")
mode = st.radio(
    "Select mode:",
    ["Ask a Question", "Evaluate a Case"],
    horizontal=True
)

st.markdown(f"**Selected Mode:** `{mode}`")
st.write("---")

# --- Input Form ---
with st.form("rag_form"):
    if mode == "Ask a Question":
        query = st.text_input(
            "💬 Enter your question:",
            placeholder="e.g., What are the main reasons for NIW denials?"
        )
    else:
        query = st.text_area(
            "🧾 Describe your case for evaluation:",
            placeholder="Provide details of your case here..."
        )

    submitted = st.form_submit_button("🚀 Run")

# --- Backend Call ---
if submitted:
    if not query.strip():
        st.warning("Please enter a question or case description before running.")
    else:
        with st.spinner("Processing your request..."):
            try:
                endpoint = f"{FASTAPI_URL}/query" if mode == "Ask a Question" else f"{FASTAPI_URL}/evaluate"
                response = requests.get(endpoint, params={"query": query})

                if response.status_code == 200:
                    result = response.json()
                    st.subheader("🧠 Response:")
                    st.write(result["response"])

                    with st.expander("📜 Full Response JSON"):
                        st.json(result)
                else:
                    st.error(f"❌ Backend returned {response.status_code}: {response.text}")
            except Exception as e:
                st.error(f"⚠️ Error connecting to backend: {e}")

# --- Footer ---
st.markdown("---")
st.caption("NIW-NP-RAG | Local RAG-powered legal reasoning system")
