from dotenv import load_dotenv
from langchain_groq import ChatGroq
def init_llama(model="llama-3.1-8b-instant"):
    load_dotenv()
    llm = ChatGroq(model=model)
    return llm

# model = init_llama()
# model.temperature = 0.7 # Temperature controls randomness in the output.
# model.top_p = 0.9 # Top-p sampling controls diversity in the output.
# # It is recommended to set temperature or top_p.
# model.max_tokens = 10 # Maximum number of tokens in the output. Specifying a max length helps you prevent long or irrelevant responses and control costs.
# model.presence_penalty = 0.5 # Presence penalty encourages the model to talk about new topics.
# model.frequency_penalty = 0.5 # Frequency penalty discourages the model from repeating the same topics.
# # It is recommended to set presence_penalty or frequency_penalty.

