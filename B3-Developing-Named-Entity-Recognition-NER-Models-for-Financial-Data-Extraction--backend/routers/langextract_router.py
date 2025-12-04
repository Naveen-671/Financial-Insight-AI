from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from services.langextract_service import LangExtractService
from config import Config

router = APIRouter()
langextract_service = LangExtractService()

class TextRequest(BaseModel):
    text: str
    models: Optional[List[str]] = None
    api_key: Optional[str] = None

class PromptUpdateRequest(BaseModel):
    prompt: str

class ExampleAddRequest(BaseModel):
    text: str
    extraction_class: str
    extraction_text: str
    attributes: Dict[str, Any]

@router.post("/extract-clauses", response_model=Dict[str, Any])
async def extract_clauses(request: TextRequest):
    """
    Extract financial and legal clauses from text using LangExtract
    
    Supports multiple models: gemini-2.5-flash-lite, gemini-2.5-pro
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Check service availability
        if not langextract_service.is_available():
            raise HTTPException(
                status_code=503, 
                detail="LangExtract service not available. Check API key configuration."
            )
        
        # Extract clauses
        result = langextract_service.extract_clauses(
            text=request.text,
            models=request.models,
            api_key=request.api_key
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Add file URLs
        if result.get("saved_files"):
            file_urls = {}
            for key, filename in result["saved_files"].items():
                if filename.endswith(".html"):
                    file_urls[key] = f"/outputs/{filename}"
                elif filename.endswith(".jsonl"):
                    file_urls[key] = f"/outputs/{filename}"
            result["file_urls"] = file_urls
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/service-status")
async def get_service_status():
    """
    Check LangExtract service status and configuration
    """
    try:
        is_available = langextract_service.is_available()
        
        return {
            "service_available": is_available,
            "google_api_key_configured": bool(Config.GOOGLE_API_KEY),
            "supported_models": Config.LANGEXTRACT_MODELS,
            "examples_count": len(langextract_service.examples),
            "recommendations": [
                "Set GOOGLE_API_KEY environment variable" if not Config.GOOGLE_API_KEY else None,
                "Install langextract package" if not is_available else None
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking service status: {str(e)}")

@router.put("/update-prompt")
async def update_extraction_prompt(request: PromptUpdateRequest):
    """
    Update the extraction prompt used for clause identification
    """
    try:
        success = langextract_service.update_prompt(request.prompt)
        
        if success:
            return {
                "success": True,
                "message": "Extraction prompt updated successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to update prompt")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating prompt: {str(e)}")

@router.post("/add-example")
async def add_training_example(request: ExampleAddRequest):
    """
    Add a new training example to improve extraction accuracy
    """
    try:
        success = langextract_service.add_example(
            text=request.text,
            extraction_class=request.extraction_class,
            extraction_text=request.extraction_text,
            attributes=request.attributes
        )
        
        if success:
            return {
                "success": True,
                "message": "Training example added successfully",
                "total_examples": len(langextract_service.examples)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add training example")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding example: {str(e)}")

@router.get("/clause-types")
async def get_supported_clause_types():
    """
    Get information about supported clause types
    """
    return {
        "supported_clause_classes": [
            "payment_clause",
            "interest_clause", 
            "termination_clause",
            "confidentiality_clause",
            "governing_law_clause",
            "force_majeure_clause",
            "prepayment_clause",
            "fee_clause",
            "dispute_resolution_clause",
            "automatic_renewal_clause"
        ],
        "common_attributes": {
            "payment_clause": ["payment_due", "late_fee", "condition"],
            "interest_clause": ["interest_rate", "calculation_period", "condition"],
            "termination_clause": ["notice_period", "triggering_party", "condition"],
            "confidentiality_clause": ["duration", "subject", "obligation"],
            "governing_law_clause": ["jurisdiction"],
            "force_majeure_clause": ["events", "effect"]
        }
    }

@router.get("/current-prompt")
async def get_current_prompt():
    """
    Get the current extraction prompt
    """
    return {
        "prompt": langextract_service.prompt,
        "examples_count": len(langextract_service.examples)
    }
