import chromadb
from chromadb.utils import embedding_functions
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Load document
with open("hr_policy.txt", "r") as f:
    text = f.read()

# ── Strategy 1: Fixed size chunking ──
def chunk_fixed(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap  # overlap keeps context between chunks
    return chunks

# ── Strategy 2: Paragraph chunking ──
def chunk_paragraphs(text):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

# ── Strategy 3: Sentence chunking ──
def chunk_sentences(text, sentences_per_chunk=3):
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    chunks = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = ". ".join(sentences[i:i+sentences_per_chunk]) + "."
        chunks.append(chunk)
    return chunks

# Compare chunk counts
fixed_chunks = chunk_fixed(text)
para_chunks = chunk_paragraphs(text)
sent_chunks = chunk_sentences(text)

# print(f"Fixed size chunks:  {len(fixed_chunks)}")
# print(f"Paragraph chunks:   {len(para_chunks)}")
# print(f"Sentence chunks:    {len(sent_chunks)}")

# print(f"\nSample fixed chunk:\n{fixed_chunks[0][:200]}")
# print(f"\nSample paragraph chunk:\n{para_chunks[0][:200]}")
# print(f"\nSample sentence chunk:\n{sent_chunks[0][:200]}")


# Use paragraph chunking — best for structured policy docs
chroma_client = chromadb.PersistentClient(path="./chroma_db_chunks")

collection = chroma_client.get_or_create_collection(
    name="hr_policy_chunks",
    embedding_function=embedding_fn
)

# Add chunks with metadata
collection.add(
    ids=[f"chunk_{i}" for i in range(len(para_chunks))],
    documents=para_chunks,
    metadatas=[{"chunk_index": i, "source": "hr_policy.txt"} for i in range(len(para_chunks))]
)

print(f"\n✓ {collection.count()} chunks stored")

# Query it
def search_and_answer(query):
    results = collection.query(
        query_texts=[query],
        n_results=2
    )
    
    context = "\n\n".join(results["documents"][0])
    
    prompt = f"""Answer the question using ONLY the information below.
If the answer isn't there, say "I don't have that information."

Context:
{context}

Question: {query}
Answer:"""

    response = client_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    print(f"\nQ: {query}")
    print(f"Retrieved chunks: {[m['chunk_index'] for m in results['metadatas'][0]]}")
    print(f"A: {response.choices[0].message.content.strip()}")

search_and_answer("How many days of annual leave do I get?")
search_and_answer("Can I fly business class for short flights?")
search_and_answer("What happens if I get a bad performance review twice?")
search_and_answer("Is dental covered in medical insurance?")