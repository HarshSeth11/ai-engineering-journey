from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import shutil
from rag_engine import (
    load_documents_from_folder,
    answer_question,
    reset_vectorstore
)

app = FastAPI(
    title="RAG API",
    description="Production-ready RAG API built with FastAPI + LangChain + ChromaDB",
    version="1.0.0"
    )

DOCUMENTS_PATH = "./documents"

# Request/Response Models
class QuestionRequest(BaseModel):
    question : str
    confidence_threshold: Optional[float] = 0.3

class AnswerResponse(BaseModel):
    answer: str
    sources: list
    chunks_used: int

# Endpoints
@app.get("/health")
def health():
    return {
        "status": "success",
        "message": "Rag Pipeline is running"
    }

@app.post("/ingest")
def ingest():
    """Load all documents from the documents/ folder"""
    results = load_documents_from_folder()
    if results["status"] == "error":
        raise HTTPException(status_code=400, detail=results["message"])
    return results

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a new .txt file"""
    if not file.filename.endswith(".txt"):
        raise HTTPException(
            status_code=400, detail="Only .txt files supported"
        )
    
    os.makedirs(DOCUMENTS_PATH, exist_ok=True)
    filepath = os.path.join(DOCUMENTS_PATH, file.filename)

    with open(filepath, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "status": "success",
        "message": f"{file.filename} uploaded. Call /ingest to process it."
    }

@app.post("/ask", response_model=AnswerResponse)
def ask(request : QuestionRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    result = answer_question(request.question, request.confidence_threshold)

    return result

@app.delete("/reset")
def reset():
    """Clear the vector store"""
    return reset_vectorstore()
