from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 1 - generate embeddings
sentences = [
    "I love dogs",
    "Puppies are adorable",
    "Cars are fast",
    "Automobiles can speed",
    "Python is a programming language",
    "I enjoy coding in Python",
]

embeddings = model.encode(sentences)
print(f"Embedding shape: {embeddings.shape}")
print(f"Each sentence → {embeddings.shape[1]} numbers")

# Step 2 - similarity matrix
similarity_matrix = cosine_similarity(embeddings)

print("\nSimilarity scores (1.0 = identical meaning):")
for i, s1 in enumerate(sentences):
    for j, s2 in enumerate(sentences):
        if i < j:
            score = similarity_matrix[i][j]
            print(f"{score:.2f} | '{s1}' vs '{s2}'")

# Step 3 - semantic search
def semantic_search(query, documents, top_k=3):
    query_embedding = model.encode([query])
    doc_embeddings = model.encode(documents)
    
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    ranked = sorted(zip(scores, documents), reverse=True)
    
    print(f"\nQuery: '{query}'")
    print("Top results:")
    for score, doc in ranked[:top_k]:
        print(f"  {score:.2f} — {doc}")

documents = [
    "Python is great for backend development",
    "Django makes web development easy",
    "Machine learning requires lots of data",
    "Neural networks power modern AI",
    "REST APIs connect frontend and backend",
    "Vector databases store embeddings efficiently",
    "LLMs can generate human-like text",
    "Transformers revolutionized natural language processing",
]

semantic_search("how do AI models work", documents)
semantic_search("web frameworks for Python", documents)
semantic_search("storing AI data", documents)
semantic_search("where are embeddings stored", documents)
semantic_search("vector storage for machine learning", documents)