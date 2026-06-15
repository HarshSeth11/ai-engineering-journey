import chromadb
from chromadb.utils import embedding_functions
from documents import documents
from query import search
from filder_by_metadata import FilterByMetadata
from delete_collection import DeleteCollection

# Setup — persists to disk automatically
client = chromadb.PersistentClient(path="./chroma_db")

# Use sentence transformers as the embedding function
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create a collection — like a table in a database
collection = client.get_or_create_collection(
    name="company_policies",
    embedding_function=embedding_fn
)

# Add documents to ChromaDB
collection.add(
    ids=[doc["id"] for doc in documents],
    documents=[doc["content"] for doc in documents],
    metadatas=[{"title": doc["title"], "category": doc["category"]} for doc in documents]
)

print(f"✓ {collection.count()} documents stored in ChromaDB")


search("How do I get my money back?", collection)
search("Is my package insured during delivery?", collection)
search("How do I secure my account?", collection)

FilterByMetadata(collection)

DeleteCollection(collection, client, embedding_fn)