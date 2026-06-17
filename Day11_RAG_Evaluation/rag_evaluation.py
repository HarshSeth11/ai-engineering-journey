from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from groq import Groq
from eval_dataset import eval_questions
import os
from dotenv import load_dotenv

load_dotenv()

# Setup RAG pipeline — same as Day 10
loader = TextLoader("hr_policy.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = splitter.split_documents(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_eval"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

prompt = PromptTemplate(
    template="""Use ONLY the context below to answer the question.
If the answer isn't in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Evaluation functions
def check_retrieval(retrieved_docs, relevant_chunks):
    """Check if relevant chunks were retrieved"""
    retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])
    hits = sum(1 for chunk in relevant_chunks if chunk.lower() in retrieved_text.lower())
    precision = hits / len(retrieved_docs)
    recall = hits / len(relevant_chunks)
    return precision, recall

def check_faithfulness(answer, context):
    """Use LLM to check if answer is faithful to context"""
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=10,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Does this answer use ONLY information from the context? Reply with just YES or NO.

Context: {context}
Answer: {answer}"""
        }]
    )
    result = response.choices[0].message.content.strip().upper()
    return 1.0 if "YES" in result else 0.0

# Run evaluation
print("Running RAG Evaluation...\n")
print(f"{'Question':<45} {'Precision':>10} {'Recall':>8} {'Faithful':>10} {'Correct':>8}")
print("-" * 85)

total_precision = total_recall = total_faithful = total_correct = 0

def check_correctness(answer, expected, question):
    """Use LLM to judge correctness instead of string matching"""
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        max_tokens=10,
        temperature=0,
        messages=[{
            "role": "user",
            "content": f"""Does this answer correctly address the question? Reply YES or NO only.

Question: {question}
Expected: {expected}
Answer: {answer}"""
        }]
    )
    result = response.choices[0].message.content.strip().upper()
    return 1.0 if "YES" in result else 0.0

for item in eval_questions:
    question = item["question"]
    expected = item["expected_answer"].lower()
    
    # Get answer and retrieved docs
    answer = chain.invoke(question)
    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)
    # Add this temporarily
    # print(f"Answer: {answer}")
    
    # Calculate metrics
    precision, recall = check_retrieval(retrieved_docs, item["relevant_chunks"])
    faithful = check_faithfulness(answer, context)
    # correct = 1.0 if expected in answer.lower() else 0.0
    correct = check_correctness(answer, expected, question)
    
    total_precision += precision
    total_recall += recall
    total_faithful += faithful
    total_correct += correct
    
    print(f"{question[:44]:<45} {precision:>10.2f} {recall:>8.2f} {faithful:>10.2f} {correct:>8.2f}")

n = len(eval_questions)
print("-" * 85)
print(f"{'AVERAGES':<45} {total_precision/n:>10.2f} {total_recall/n:>8.2f} {total_faithful/n:>10.2f} {total_correct/n:>8.2f}")