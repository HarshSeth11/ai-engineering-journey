from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import numpy as np
from .documents import documents
from dotenv import load_dotenv
import os

# Load variables from the .env file
load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Embed documents once at startup
doc_embeddings = model.encode([doc["content"] for doc in documents])

def retrieve(query, top_k=2, threshold=0.3):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    ranked = sorted(zip(scores, documents), reverse=True)
    return [(score, doc) for score, doc in ranked[:top_k] if score >= threshold]

def answer_question(query):
    results = retrieve(query)
    
    if not results:
        return {
            "answer": "I don't have relevant information to answer this.",
            "sources": [],
            "retrieved_count": 0
        }
    
    context = "\n\n".join([
        f"[{doc['title']}]\n{doc['content']}"
        for score, doc in results
    ])
    
    prompt = f"""Answer the user's question using ONLY the information provided below.
If the answer is not in the provided information, say "I don't have that information."
Do not make up any details.

Information:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    return {
        "answer": response.choices[0].message.content.strip(),
        "sources": [doc["title"] for _, doc in results],
        "retrieved_count": len(results)
    }