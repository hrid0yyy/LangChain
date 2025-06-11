from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
llm = ChatMistralAI(
    model="mistral-large-latest"
)
embeddings = MistralAIEmbeddings(model="mistral-embed")
