from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

def init_llm():
    """Initialize and return the ChatGoogleGenerativeAI model."""
    load_dotenv()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",)
    return llm

def call_llm(prompt):
    """Call the LLM with the given prompt and return the response content."""
    llm = init_llm()
    response = llm.invoke(prompt)
    return response.content