import argparse
import json
import re
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import fitz  # PyMuPDF
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_agent
from niw_np_rag.config.config import GOOGLE_API_KEY
import datetime
import os

from langchain_huggingface.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",      # Uses GPU if available
    torch_dtype="auto",     # FP16/FP32 based on hardware
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.9,
)

local_llm = HuggingFacePipeline(pipeline=pipe)


# ------------------ System Prompt ------------------
SYSTEM_PROMPT_QUERY = (
    "You are an expert legal assistant specializing in EB2 National Interest Waiver (NIW) petitions. "
    "Your task is to summarize and explain the information found in the retrieved context, "
    "strictly based on the provided text without adding any external knowledge, opinions, or interpretations.\n\n"
    "When referring to evidence, always cite your sources clearly by including the document title, "
    "case identifier, or source URL each time you reference contextual information.\n\n"
)

# ------------------ Question–Answer Generation Prompt ------------------
QA_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["system_prompt", "text"],
    template = (
        "You are an expert assistant specialized in generating question–answer pairs for EB2 National Interest Waiver (NIW) applicants.\n\n"
        "Your role is to analyze the given AAO case text and create realistic, diverse, and informative question–answer pairs "
        "that reflect the kinds of questions prospective EB2 NIW applicants might ask when preparing their petitions.\n\n"
        "Instructions:\n"
        "- Use only the information available in the provided case text.\n"
        "- Generate up to 5 unique and well-phrased question–answer pairs.\n"
        "- Format your output exactly as follows:\n\n"
        "question: <question>\n"
        "answer: <answer>\n\n"
        "Case Text:\n{text}"
    ),
)

# Extract the data from the text: For example, 
extract_prompt = (
    "Extract key metadata from the following AAO NIW case:"
    "- case_id"
    "- decision (approved / dismissed / sustained / denied)"
    "- date (if present)"
    "- short reason or summary"
    "Return as JSON only."
    "Text:{chunk}"
)

# ------------------ Helper: Filter PDFs by Date ------------------
def filter_pdfs_by_date(pdf_files, cutoff_date=datetime.datetime(2016, 12, 27)):
    filtered_pdfs = []
    for pdf_file in pdf_files:
        date_str = os.path.basename(pdf_file).split('_')[0]
        file_date = datetime.datetime.strptime(date_str, '%b%d%Y')
        if file_date > cutoff_date:
            filtered_pdfs.append(pdf_file)
    return filtered_pdfs

# ------------------ Helper: Extract Text ------------------
def extract_text_from_pdf(path: Path) -> str:
    """Extract and clean text from a PDF file."""
    with fitz.open(str(path)) as doc:
        text = "\n".join([page.get_text("text") for page in doc])
    return re.sub(r"\s+", " ", text).strip()

# ------------------ Main QA Generation ------------------
def generate_qna_from_text(text: str, llm) -> List[Dict[str, str]]:
    """Generate multiple question–answer pairs from input text using LLM."""
    prompt = QA_PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT_QUERY, text=text)  # limit for context
    response = llm.invoke(prompt)
    response = response.content if hasattr(response, "content") else str(response)
    print("Generated Response:\n", response)  # Debug print

    # Parse "question:" and "answer:" pairs
    qna_pairs = []
    qa_blocks = re.split(r"(?=question:)", response, flags=re.IGNORECASE)
    for block in qa_blocks:
        q_match = re.search(r"question:\s*(.*)", block, flags=re.IGNORECASE)
        a_match = re.search(r"answer:\s*(.*)", block, flags=re.IGNORECASE)
        if q_match and a_match:
            qna_pairs.append({"question": q_match.group(1).strip(), "answer": a_match.group(1).strip()})
    return qna_pairs

# ------------------ Entry Point ------------------
def main(pdf_dir: str, output_path: str):
    """Generate question–answer dataset from all PDFs in a directory."""
    # llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", google_api_key=GOOGLE_API_KEY)
    llm = local_llm  # Use local LLM for generation
    dataset = []
    pdf_files = filter_pdfs_by_date(list(Path(pdf_dir).glob("*.pdf")))
    print(Path(pdf_dir))
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        text = extract_text_from_pdf(pdf_file)
        print(f"Processing {pdf_file.name}, extracted {len(text)} characters of text.")
        qna_pairs = generate_qna_from_text(text, llm)
        for item in qna_pairs:
            dataset.append({
                "source": pdf_file.name,
                "question": item["question"],
                "answer": item["answer"]
            })
        extract = 
        break

    # Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # with open(output_path, "w", encoding="utf-8") as f:
    #     json.dump(dataset, f, indent=2, ensure_ascii=False)

    # print(f"✅ Saved {len(dataset)} question–answer pairs to {output_path}")

# ------------------ CLI ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate question–answer pairs from NIW PDFs.")
    parser.add_argument("--pdf_dir", type=str, required=True, help="Path to directory containing PDFs")
    parser.add_argument("--output", type=str, default="./niw_qna.json", help="Output JSON file path")
    args = parser.parse_args()

    main(args.pdf_dir, args.output)
