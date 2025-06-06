from dotenv import load_dotenv
from langchain_groq import ChatGroq
def init_llm():
    """Initialize and return the ChatGoogleGenerativeAI model."""
    load_dotenv()
    llm = ChatGroq(model="llama-3.1-8b-instant",)
    return llm

def call_llm(prompt):
    """Call the LLM with the given prompt and return the response content."""
    llm = init_llm()
    response = llm.invoke(prompt)
    return response.content


print(call_llm("What is the capital of France?"))