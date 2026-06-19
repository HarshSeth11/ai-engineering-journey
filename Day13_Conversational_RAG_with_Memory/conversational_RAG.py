import shutil
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# ── Setup RAG pipeline ──
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
    
    if os.path.exists("./chroma_conv"):
        shutil.rmtree("./chroma_conv")
    
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_conv"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 2})

retriever = load_and_store()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

# ── Step 1: Question rewriter ──
# Rewrites vague follow-up questions into standalone questions
rewriter_prompt = ChatPromptTemplate.from_messages([
    ("system", """Your job is to rewrite questions only. 
Do NOT answer the question.
Do NOT add information from chat history into the rewrite.
Simply make the question standalone by replacing pronouns with their referents.

Example:
History: Q: How many leave days? A: 20 days
Follow-up: "Can I carry them forward?"
Rewrite: "Can annual leave days be carried forward?"

Return ONLY the rewritten question. Nothing else."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Rewrite this question as standalone: {question}")
])

rewriter = rewriter_prompt | llm | StrOutputParser()

# ── Step 2: Answer chain ──
answer_prompt = PromptTemplate(
    template="""Answer the question using ONLY the context below.
Mention the source document in your answer.
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

# ── Conversational RAG function ──
chat_history = []

def chat(question):
    # Step 1 — rewrite question if it's a follow-up
    if chat_history:
        standalone_question = rewriter.invoke({
            "chat_history": chat_history,
            "question": question
        })
        print(f"🔄 Rewritten: {standalone_question}")
    else:
        standalone_question = question
    
    # Step 2 — retrieve relevant chunks
    docs = retriever.invoke(standalone_question)
    context = format_docs(docs)
    
    # Step 3 — generate answer
    filled_prompt = answer_prompt.format(
        context=context,
        question=standalone_question
    )
    answer = llm.invoke(filled_prompt).content
    
    # Step 4 — update chat history
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=answer))
    
    print(f"\nQ: {question}")
    print(f"A: {answer}")
    print(f"📄 Sources: {list(set([d.metadata.get('source_name') for d in docs]))}")
    print("-" * 60)

# ── Test conversation ──
print("=== Conversation 1: Leave Policy ===")
chat("How many days of annual leave do I get?")
chat("Can I carry them forward?")          # tests memory — "them" = leave days
chat("What about sick leave?")             # tests memory — "what about" implies context

print("\n=== Conversation 2: Benefits ===")
chat_history.clear()                       # start fresh conversation
chat("Is dental covered in insurance?")
chat("What about vision?")                 # tests memory
chat("How much does the company pay?")     # tests memory — "how much" for what?