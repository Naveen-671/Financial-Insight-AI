# routers/rag_router.py
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import Optional
from pydantic import BaseModel
import os
from pathlib import Path
from services.rag_service import rag_service, initialize_rag_service
from config import Config

router = APIRouter(prefix="/api/v1/rag", tags=["RAG"])

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3

@router.on_event("startup")
async def startup_event():
    """Initialize RAG service on startup."""
    try:
        pinecone_key = os.getenv("PINECONE_API_KEY")
        success = initialize_rag_service(Config.GOOGLE_API_KEY, pinecone_key, Config.GROQ_API_KEY)
        if success:
            print("RAG service initialized successfully")
        else:
            print("Warning: RAG service initialization failed")
    except Exception as e:
        print(f"RAG startup error: {e}")

@router.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the RAG system.
    Supports PDF, TXT, and DOCX files.
    """
    try:
        # Validate file type
        allowed_extensions = {".pdf", ".txt", ".docx", ".doc"}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}"
            )
        
        # Save uploaded file temporarily
        upload_dir = Config.UPLOADS_DIR
        file_path = upload_dir / file.filename
        
        content = await file.read()
        file_path.write_bytes(content)
        
        # Process document
        success = rag_service.process_document(file_path)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process document")
        
        return {
            "success": True,
            "message": f"Document '{file.filename}' uploaded and processed successfully",
            "filename": file.filename,
            "size_bytes": len(content)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@router.post("/query")
async def rag_query(request: QueryRequest):
    """
    Query the RAG system with a question.
    """
    try:
        return rag_service.query(request.question, top_k=request.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status")
async def get_rag_status():
    """Check if RAG service is initialized and get stats."""
    stats = {
        "initialized": rag_service.initialized,
        "using_pinecone": rag_service.index is not None,
        "google_api_configured": bool(rag_service.api_key),
    }
    
    # Add document count
    if rag_service.index:
        stats["backend"] = "Pinecone"
    else:
        stats["backend"] = "In-Memory"
        stats["total_documents"] = len(rag_service.documents)
    
    return stats

@router.delete("/clear")
async def clear_documents():
    """Clear all documents from RAG system."""
    try:
        if rag_service.index:
            # Clear Pinecone index
            # Note: delete_all is not supported in all Pinecone environments
            rag_service.index.delete(delete_all=True)
            message = "Cleared all vectors from Pinecone"
        else:
            # Clear in-memory
            rag_service.documents.clear()
            message = "Cleared all in-memory documents"
        
        return {"success": True, "message": message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))