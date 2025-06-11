from src.loader import load_pdf
from src.splitter import split_docs
from src.vector_store import create_vector_store, load_vector_store, vector_store_exists
from src.retriever import retriever
from src.prompt import get_prompt
from langchain_mistralai import ChatMistralAI
try:
    # Load and process PDF
    docs = load_pdf("HowToDevelop_WBS.pdf")
    if not docs:
        raise ValueError("No documents loaded from PDF")

    chunks = split_docs(docs)
    if not chunks:
        raise ValueError("No chunks created from documents")

    # Create vector store if it doesn't exist
    if not vector_store_exists():
        create_vector_store(chunks)

    # Load vector store
    vector_store = load_vector_store()
    llm = ChatMistralAI(
    model="mistral-large-2407",
    )
    retriever_instance = retriever(vectorstore=vector_store, llm = llm)
    # Print results
    docs = retriever_instance.invoke("What is the purpose of a WBS?")
    context = " ".join([doc.page_content for doc in docs])
    question = "What is the purpose of a WBS?"
    prompt = get_prompt(context, question)
    print(llm.invoke(prompt))

except Exception as e:
    print(f"Error: {str(e)}")