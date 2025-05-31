# Import necessary LangChain and Hugging Face modules
from langchain_huggingface import HuggingFaceEndpoint  # Connects to Hugging Face's Inference API for open-source LLMs
from langchain.prompts import PromptTemplate          # Creates reusable prompt templates with dynamic inputs
from langchain_core.runnables import RunnableSequence # Chains components (prompt and LLM) for processing inputs
from dotenv import load_dotenv                       # Loads environment variables from a .env file

# Load environment variables from a .env file
load_dotenv()  # Retrieves sensitive data like HUGGINGFACEHUB_API_TOKEN from .env file

# Initialize the Hugging Face LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",  # Specifies the open-source model (Mixtral, instruction-tuned)
    temperature=0.9,                                 # Sets randomness of output (0=deterministic, 1=creative)
    max_new_tokens=100                              # Limits response to 100 new tokens for concise output
)  # Uses API key from environment variable HUGGINGFACEHUB_API_TOKEN

# Define a prompt template for structuring the input
prompt = PromptTemplate(
    input_variables=["question"],                    # Placeholder variable for the user's input
    template="Answer the following question in a concise manner: {question}"  # Formats the input question
)  # Creates a prompt like: "Answer the following question in a concise manner: What is the capital of France?"

# Create a chain to process the prompt and LLM together
chain = RunnableSequence(prompt | llm)  # Chains prompt formatting and LLM invocation using the | operator

# Define the input question
question = "What is the capital of France?"  # Sample question to test the chain

# Run the chain with the input question
response = chain.invoke({"question": question})  # Formats prompt, sends to LLM, and gets response

# Print the LLM's response
print(response)  # Outputs the answer, e.g., "The capital of France is Paris."