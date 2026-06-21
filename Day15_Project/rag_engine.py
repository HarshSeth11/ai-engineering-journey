import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()

CHROMA_PATH = "./chroma_fastapi"
DOCUMENTS_PATH = "./documents"

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
llm_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

vectorstore = None
all_chunks = []

def load_documents_from_folder():
    """Load all .txt files from document folder"""
    global vectorstore, all_chunks

    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
        except PermissionError:
            # Force garbage collection to release file handles
            import gc
            gc.collect()
            shutil.rmtree(CHROMA_PATH)

    all_docs = []
    loaded_files = []

    for filename in os.listdir(DOCUMENTS_PATH):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCUMENTS_PATH, filename)
            loader = TextLoader(filepath)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source_name"] = filename.replace(".txt", "")
            all_docs.extend(docs)
            loaded_files.append(filename)
    
    if not all_docs:
        return {"status": "error", "message" : "No documents found"}
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    all_chunks = splitter.split_documents(all_docs)

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    
    return {
        "status": "success",
        "files_loaded": loaded_files,
        "chunks created": len(all_chunks)
    }

def keyword_search(query, k=4):
    query_words = set(query.lower().split())
    scored = []
    for chunk in all_chunks:
        chunk_words = set(chunk.page_content.lower().split())
        intersaction = query_words & chunk_words
        union = query_words | chunk_words
        score = len(intersaction)/len(union) if union else 0
        scored.append((score, chunk))

    scored.sort(key=lambda x : x[0], reverse=True)
    return [chunk for score, chunk in scored[:k] if score > 0]

def hybrid_search(query, k=4):
    if not vectorstore:
        return []
    
    sementic_docs = vectorstore.as_retriever(
        search_kwargs={"k":k}
    ).invoke(query)

    keyword_docs = keyword_search(query, k)

    seen = set()
    combine = []

    for doc in sementic_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            combine.append(("sementic", doc))
        
    for doc in keyword_docs:
        key = doc.page_content[:100]
        if key not in seen:
            seen.add(key)
            combine.append(("keyword", doc))
    
    return combine

def rerank(query, candidates, top_k=2):
    if not candidates:
        return []
    
    top_sementic = next(((s,d) for s,d in candidates if s=="sementic"), None)

    try:
        candidate_text = ""
        for i, (source, doc) in enumerate(candidates):
            candidate_text += f"[{i}] ({source}): {doc[:200]}\n\n"

        response = llm_groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=100,
            temperature=0,
            messages=[{
                "role" : "user",
                "content" : f"""Given this query: "{query}"
                Rank these chunks by relevence. Return Only JSON array of indices.
                Example: [2, 0, 1, 3]
                
                Chunks: {candidate_text}

Return JSON array only:"""
            }]
        )

        raw = response.choices[0].message.content.strip()
        indices = json.loads(raw)
        reranked = [candidates[i] for i in indices[:top_k] if i < len(candidates)]

        if top_sementic and top_sementic not in reranked:
            reranked = [top_sementic] + reranked[:top_k-1]
        
        return reranked
    except:
        return candidates[:top_k]
    
def answer_question(query: str, confidence_threshold: float = 0.3):
    """Main RAG function - return answer with source"""

    if not vectorstore:
        return {
            "answer" : "No document loaded. Please ingest document first.",
            "source" : [],
            "chunks_used": 0
        }

    candidates = hybrid_search(query)

    # confidence check on top sementic seach
    sementic_result = vectorstore.similarity_search_with_score(query, k=1)
    if sementic_result:
        top_score = sementic_result[0][1]
        if top_score > (1-confidence_threshold):
            return {"answer": "I don't have relevant information to answer this question.",
                "sources": [],
                "chunks_used": 0
                }
        
    reranked = rerank(query, candidates, top_k=2)

    if not reranked:
        return {"answer": "I don't have relevant information to answer this question.",
                "sources": [],
                "chunks_used": 0
                }

    context = "\n\n".join([f"[Source: {doc.metadata.get('source_name')}]\n{doc.page_content}" for _, doc in reranked])

    prompt = f"""Answer using ONLY the context below.
Mention which document your answer comes from.
If not available say "I don't have that information."

Context:
{context}

Question: {query}
Answer:"""
    
    response = llm_groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=300,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip()
    sources = list(set([doc.metadata.get("source_name") for _, doc in reranked]))

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(reranked)
    }

def reset_vectorstore():
    """Clear everything"""
    global all_chunks, vectorstore
    all_chunks = []
    vectorstore = None
    
    import gc
    gc.collect()
    import time
    time.sleep(1)
    
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            return {"status": "success", "message": "Vector store cleared"}
        except PermissionError:
            # On Windows, rename instead of delete
            try:
                old_path = CHROMA_PATH + "_old"
                os.rename(CHROMA_PATH, old_path)
                return {
                    "status": "success", 
                    "message": "Vector store cleared. Old files renamed for cleanup."
                }
            except Exception as e:
                return {
                    "status": "success",
                    "message": "Vector store cleared from memory. Restart server to fully clean disk."
                }
    
    return {"status": "success", "message": "Vector store cleared"}