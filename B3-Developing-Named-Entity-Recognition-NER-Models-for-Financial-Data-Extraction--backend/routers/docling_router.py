from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
import os
from services.docling_service import DoclingService
from config import Config

router = APIRouter()
docling_service = DoclingService()

@router.post("/extract-text", response_model=Dict[str, Any])
async def extract_text_from_document(file: UploadFile = File(...)):
    """
    Extract text from uploaded document using Docling
    
    Supports: PDF, TXT, HTML, DOCX, DOC
    """
    try:
        # Validate file
        file_content = await file.read()
        validation_result = docling_service.validate_file(file.filename, len(file_content))
        
        if not validation_result["valid"]:
            raise HTTPException(status_code=400, detail=validation_result["error"])
        
        # Save uploaded file
        file_path = docling_service.save_uploaded_file(file_content, file.filename)
        
        try:
            # Extract text
            result = docling_service.extract_text_from_file(file_path)
            
            if not result["success"]:
                raise HTTPException(status_code=500, detail=result["error"])
            
            # Add file info to response
            result["file_info"] = {
                "original_filename": file.filename,
                "saved_path": file_path
            }
            
            return JSONResponse(content=result)
            
        finally:
            # Cleanup uploaded file
            docling_service.cleanup_file(file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get list of supported file formats
    """
    return {
        "supported_extensions": list(Config.ALLOWED_EXTENSIONS),
        "max_file_size_mb": Config.MAX_FILE_SIZE / (1024 * 1024)
    }

@router.post("/extract-from-path")
async def extract_text_from_path(file_path: str):
    """
    Extract text from file path (for testing/local files)
    """
    try:
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
        
        result = docling_service.extract_text_from_file(file_path)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return JSONResponse(content=result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
