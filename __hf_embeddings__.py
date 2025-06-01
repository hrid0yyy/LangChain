import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the locally saved model
model = SentenceTransformer("D:/LangChain/models/all-MiniLM-L6-v2")


# Sentences to embed
sentences = [
    "That is a happy person",
    "That is a happy dog",
    "That is a very happy person",
    "Today is a sunny day"
]

#  Encode sentences into embeddings
embeddings = model.encode(sentences)

#  Compute cosine similarity matrix
similarities = cosine_similarity(embeddings)

#  Print result
print("Similarity matrix shape:", similarities.shape)
print(np.round(similarities, 2))  # Optional: rounded for readability
