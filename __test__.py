
from gemini import init_gemini
from embedding_models import embeddings
model = init_gemini()

print(model.invoke("What is the capital of France?").content)
#print(embeddings.embed_query("Hello, world!"))