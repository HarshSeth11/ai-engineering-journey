from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import numpy as np
from documents import documents
import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

doc_embeddings = model.encode([doc["content"] for doc in documents])

query_embedding = model.encode(["How long do I have to return a product?"])
print("\n\nQuery Embedding : ", query_embedding)
result = cosine_similarity(query_embedding, doc_embeddings)[0]

ranked = sorted(zip(result, documents), reverse=True)
print("\n\nRanked : ", ranked)

print("Full result shape:", result.shape)
print("Full result:", result)
print("After [0]:", result[0])