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

# Step 1 — embed all documents at startup
print("Embedding documents...")
doc_embeddings = model.encode([doc["content"] for doc in documents])
print(f"✓ {len(documents)} documents embedded\n")

def retrieve(query, top_k=2, threshold=0.3):
    """Find most relevant documents for a query"""
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, doc_embeddings)[0]
    ranked = sorted(zip(scores, documents), reverse=True)
    return [(score, doc) for score, doc in ranked[:top_k] if score>=threshold]

def answer(query):
    # Retrieve relevant docs
    results = retrieve(query)
    
    # Build context from retrieved docs
    context = "\n\n".join([
        f"[{doc['title']}]\n{doc['content']}"
        for score, doc in results
    ])

    print("\n\n\nContext: ", context)
    print("\n\n\n")
    
    # Ask LLM to answer using only the context
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
    
    answer_text = response.choices[0].message.content.strip()
    
    # Show what was retrieved
    print(f"Q: {query}")
    print(f"📄 Retrieved: {[doc['title'] for _, doc in results]}")
    print(f"A: {answer_text}\n")

# Test it
# answer("How long do I have to return a product?")
# answer("What's the cost of express shipping?")
# answer("How many users does the Pro plan support?")
# answer("What is the CEO's name?")  # should say it doesn't know

answer("What is the capital of France?")  # completely unrelated to your docs
answer("How long do I have to return a product?")  # should still work