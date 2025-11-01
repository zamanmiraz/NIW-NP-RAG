from niw_np_rag.app.llm_rag import LLMRAG

def test_llm_rag_initialization():
    llm_rag = LLMRAG()
    print("LLMRAG initialized successfully.")
    response = llm_rag.generate_response("Could you please tell me about the in which different fields the petitioner have been working")
    print("LLMRAG initialized and response generated successfully.")
    print("Response:", response)

if __name__ == "__main__":
    test_llm_rag_initialization()