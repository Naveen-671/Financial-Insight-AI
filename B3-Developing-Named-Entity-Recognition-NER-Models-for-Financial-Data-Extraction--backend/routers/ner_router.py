from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from services.ner_service import NERService
from config import Config

router = APIRouter()
ner_service = NERService()

class TextRequest(BaseModel):
    text: str
    model_path: Optional[str] = None

class TrainRequest(BaseModel):
    annotations_file: str
    output_dir: str

@router.post("/extract-entities", response_model=Dict[str, Any])
async def extract_entities(request: TextRequest):
    """
    Extract named entities from text using trained NER model
    
    Entity types include: COMPANY, PERSON, MONEY, DATE, etc.
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Load model if specified
        if request.model_path:
            if not ner_service.load_model(request.model_path):
                raise HTTPException(status_code=500, detail="Failed to load specified NER model")
        
        # Extract entities
        result = ner_service.extract_entities(request.text)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Add statistics
        result["statistics"] = ner_service.get_entity_statistics(result["entities"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/model-status")
async def get_model_status():
    """
    Check NER model status and availability
    """
    try:
        is_available = ner_service.is_model_available()
        
        return {
            "model_loaded": is_available,
            "inference_method": "huggingface_api" if ner_service.hf_api_key else "regex_fallback",
            "hf_model": ner_service.hf_model,
            "api_key_configured": bool(ner_service.hf_api_key),
            "recommendations": [
                "Set HUGGINGFACE_API_KEY for better NER accuracy" if not ner_service.hf_api_key else None
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking model status: {str(e)}")

@router.post("/load-model")
async def load_ner_model(model_path: Optional[str] = None):
    """
    Load NER model from specified path or default config path
    """
    try:
        success = ner_service.load_model(model_path)
        
        if success:
            return {
                "success": True,
                "message": "NER model loaded successfully",
                "model_path": model_path or str(Config.NER_MODEL_PATH)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load NER model")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@router.post("/train-model")
async def train_ner_model(request: TrainRequest):
    """
    Train NER model from annotations file.
    Note: Training is not available in API-based deployment mode.
    """
    return {
        "success": False,
        "error": "Model training is not available in API-based deployment mode. Use local development environment for training."
    }

@router.get("/entity-types")
async def get_supported_entity_types():
    """
    Get information about supported entity types
    """
    return {
        "common_entity_types": [
            "COMPANY", "PERSON", "MONEY", "DATE", "TIME",
            "PERCENT", "FACILITY", "GPE", "PRODUCT", "EVENT"
        ],
        "financial_entity_types": [
            "COMPANY", "MONEY", "PERCENT", "DATE", "CONTRACT",
            "AGREEMENT", "LOAN", "INTEREST_RATE", "PAYMENT"
        ],
        "note": "Actual entity types depend on your trained model"
    }
