import shutil
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# Step 1 — Load multiple documents with source tracking
def load_documents():
    all_docs = []
    
    sources = [
        ("hr_policy.txt", "HR Policy"),
        ("company_handbook.txt", "Company Handbook"),
        ("benefits_guide.txt", "Benefits Guide"),
    ]
    
    for filename, source_name in sources:
        loader = TextLoader(filename)
        docs = loader.load()
        
        # Tag each document with its source name
        
        for doc in docs:
            doc.metadata["source_name"] = source_name
            doc.metadata["filename"] = filename
        
        all_docs.extend(docs)
        print(f"✓ Loaded {source_name}")
    
    return all_docs


# Step 2 — Chunk with source metadata preserved
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks across all documents")
    return chunks

# Step 3 — Store in ChromaDB
def create_vectorstore(chunks):
    if os.path.exists("./chroma_multi"):
        shutil.rmtree("./chroma_multi")
    
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_multi"
    )
    print(f"✓ Stored in ChromaDB")
    return vectorstore

# Setup
documents = load_documents()
chunks = chunk_documents(documents)
vectorstore = create_vectorstore(chunks)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

prompt = PromptTemplate(
    template="""Answer the question using ONLY the context below.
Always mention which document your answer comes from.
If the answer isn't available, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source_name", "Unknown")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n".join(formatted)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Ask with source citations
def ask(question):
    answer = chain.invoke(question)
    docs = retriever.invoke(question)
    sources = list(set([doc.metadata.get("source_name") for doc in docs]))
    
    print(f"\nQ: {question}")
    print(f"A: {answer}")
    print(f"📄 Sources searched: {sources}")
    print("-" * 60)

ask("How many days of annual leave do I get?")
ask("What is the dress code on Fridays?")
ask("Does the company match 401k contributions?")
ask("What happens if I violate confidentiality?")
ask("Is dental covered and how much wellness allowance do I get?")