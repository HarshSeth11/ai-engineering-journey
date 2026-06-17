from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

# Step 1 — Load document
loader = TextLoader("hr_policy.txt")
documents = loader.load()
print(f"✓ Loaded {len(documents)} document")

# Step 2 — Chunk it
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(documents)
print(f"✓ Split into {len(chunks)} chunks")

# Step 3 — Embed and store in ChromaDB
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_langchain"
)
print(f"✓ Stored in ChromaDB")

# Step 4 — Build retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

# Step 5 — Custom prompt
prompt = PromptTemplate(
    template="""Use ONLY the context below to answer the question.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# Step 6 — LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# Step 7 — Chain everything together
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def ask(question):
    result = chain.invoke(question)
    docs = retriever.invoke(question)
    print(f"\nQ: {question}")
    print(f"A: {result}")
    print(f"Sources: {[doc.metadata for doc in docs]}")

ask("How many days of annual leave do I get?")
ask("Can I fly business class?")
ask("Is dental covered in insurance?")
ask("What is the maximum hotel reimbursement per night?")