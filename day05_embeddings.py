import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import os
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_embedding(text):
    """Get embedding via LLM trick — ask the model to represent meaning as scores"""
    prompt = f"""Rate how strongly this text relates to each of these 10 concepts.
Return ONLY a JSON array of 10 float numbers between 0 and 1.
No explanation, no markdown, just the raw JSON array.

Concepts: technology, nature, emotion, action, time, space, quality, quantity, abstract, concrete

Text: "{text}"

Example output: [0.8, 0.1, 0.2, 0.5, 0.3, 0.1, 0.7, 0.2, 0.4, 0.6]"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=100,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)

def semantic_search(query, documents, top_k=3):
    print(f"\nGenerating embeddings...")
    
    query_embedding = np.array(get_embedding(query)).reshape(1, -1)
    doc_embeddings = np.array([get_embedding(doc) for doc in documents])
    
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    ranked = sorted(zip(scores, documents), reverse=True)
    
    print(f"Query: '{query}'")
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