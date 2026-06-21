import shutil
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from groq import Groq
import json

load_dotenv()

# ── Setup ──
def load_and_store():
    sources = [
        ("hr_policy.txt", "HR Policy"),
        ("company_handbook.txt", "Company Handbook"),
        ("benefits_guide.txt", "Benefits Guide"),
    ]
    
    all_docs = []
    for filename, source_name in sources:
        loader = TextLoader(filename)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_name"] = source_name
        all_docs.extend(docs)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(all_docs)
    
    if os.path.exists("./chroma_hybrid"):
        shutil.rmtree("./chroma_hybrid")
    
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_hybrid"
    )
    return vectorstore, chunks

vectorstore, all_chunks = load_and_store()
print(f"✓ Loaded {len(all_chunks)} chunks")

llm_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# ── Retrieval Method 1: Semantic Search ──
def semantic_search(query, k=4):
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever.invoke(query)

# ── Retrieval Method 2: Keyword Search ──
def keyword_search(query, chunks, k=4):
    """Simple BM25-style keyword matching"""
    query_words = set(query.lower().split())
    
    scored = []
    for chunk in chunks:
        chunk_words = set(chunk.page_content.lower().split())
        # Jaccard similarity — overlap between query and chunk words
        intersection = query_words & chunk_words
        union = query_words | chunk_words
        score = len(intersection) / len(union) if union else 0
        scored.append((score, chunk))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:k] if score > 0]

# ── Retrieval Method 3: Hybrid Search ──
def hybrid_search(query, k=4, semantic_weight=0.7, keyword_weight=0.3):
    """Combine semantic and keyword search results"""
    semantic_docs = semantic_search(query, k=k)
    keyword_docs = keyword_search(query, all_chunks, k=k)
    
    # Merge results — deduplicate by content
    seen = set()
    combined = []
    
    for doc in semantic_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            combined.append(("semantic", doc))
    
    for doc in keyword_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            combined.append(("keyword", doc))
    
    return combined

# ── Reranker ──
# def rerank(query, candidates, top_k=2):
    """Use LLM to rerank candidates by relevance"""
    if not candidates:
        return []
    
    # Build candidate list for LLM
    candidate_text = ""
    for i, (source, doc) in enumerate(candidates):
        candidate_text += f"[{i}] ({source}): {doc.page_content[:200]}\n\n"
    
    response = llm_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=100,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Given this query: "{query}"
            
Rank these chunks by relevance. Return ONLY a JSON array of indices in order of relevance.
Example: [2, 0, 3, 1]

Chunks:
{candidate_text}

Return JSON array only:"""
        }]
    )
    
    try:
        raw = response.choices[0].message.content.strip()
        indices = json.loads(raw)
        reranked = [candidates[i] for i in indices[:top_k] if i < len(candidates)]
        return reranked
    except:
        return candidates[:top_k]

def rerank(query, candidates, top_k=2):
    if not candidates:
        return []
    
    # Always keep the top semantic result as an anchor
    top_semantic = next(((s,d) for s,d in candidates if s=="semantic"), None)
    
    try:
        candidate_text = ""
        for i, (source, doc) in enumerate(candidates):
            candidate_text += f"[{i}] ({source}): {doc.page_content[:200]}\n\n"
        
        response = llm_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=100,
            temperature=0,
            messages=[{
                "role": "user",
                "content": f"""Given this query: "{query}"
Rank these chunks by relevance. Return ONLY a JSON array of indices.
Example: [2, 0, 3, 1]

Chunks:
{candidate_text}

Return JSON array only:"""
            }]
        )
        
        raw = response.choices[0].message.content.strip()
        indices = json.loads(raw)
        reranked = [candidates[i] for i in indices[:top_k] if i < len(candidates)]
        
        # If top semantic result got dropped, add it back
        if top_semantic and top_semantic not in reranked:
            reranked = [top_semantic] + reranked[:top_k-1]
        
        return reranked
        
    except:
        return candidates[:top_k]

# ── Final answer ──
def answer_with_hybrid_rag(query):
    print(f"\nQ: {query}")
    
    # Get candidates via hybrid search
    candidates = hybrid_search(query, k=4)
    print(f"📥 Retrieved {len(candidates)} candidates ({sum(1 for s,_ in candidates if s=='semantic')} semantic, {sum(1 for s,_ in candidates if s=='keyword')} keyword)")
    
    print("Candidates before rerank:")
    for i, (source, doc) in enumerate(candidates):
        print(f"  [{i}] ({source}): {doc.page_content[:80]}")

    # Rerank
    reranked = rerank(query, candidates, top_k=2)
    print(f"🎯 Reranked to top {len(reranked)}")
    
    # Build context
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source_name')}]\n{doc.page_content}"
        for _, doc in reranked
    ])
    
    # Generate answer
    prompt = f"""Answer using ONLY the context below. 
Mention the source document.
If not available say "I don't have that information."

Context:
{context}

Question: {query}
Answer:"""
    
    response = llm_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=200,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    
    answer = response.choices[0].message.content.strip()
    sources = list(set([doc.metadata.get("source_name") for _, doc in reranked]))
    
    print(f"A: {answer}")
    print(f"📄 Sources: {sources}")
    print("-" * 60)

# ── Test it ──
answer_with_hybrid_rag("How many days of annual leave do I get?")
answer_with_hybrid_rag("What is the dress code policy?")
answer_with_hybrid_rag("401k matching details")        # keyword-heavy query
answer_with_hybrid_rag("What are my mental health benefits?")
answer_with_hybrid_rag("NDA requirements for employees")  # exact term match