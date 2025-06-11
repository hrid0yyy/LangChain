from rag_qna import RAGQNA
from langchain_mistralai import ChatMistralAI

try:
    # Initialize RAGQNA with a PDF
    
    llm = ChatMistralAI(
        model="mistral-large-2407",
    )
    rag = RAGQNA(pdf_path="HowToDevelop_WBS.pdf", llm=llm)

    # Query for relevant documents
    question = "What is the purpose of a WBS?"
    docs = rag.query(question)
    for doc in docs:
        print(doc.page_content)

    # Get a formatted answer
    answer = rag.answer(question)
    print(answer)

except Exception as e:
    print(f"Error: {str(e)}")