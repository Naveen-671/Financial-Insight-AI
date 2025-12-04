from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from services.finbert_service import FinBERTService
from config import Config

router = APIRouter()
finbert_service = FinBERTService()

class TextRequest(BaseModel):
    text: str
    model_name: Optional[str] = None

class HTMLRequest(BaseModel):
    html_content: str
    model_name: Optional[str] = None

@router.post("/analyze-sentiment", response_model=Dict[str, Any])
async def analyze_sentiment(request: TextRequest):
    """
    Analyze sentiment of text using FinBERT
    
    Returns sentiment label (positive/negative/neutral) and confidence score
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Analyze sentiment
        result = finbert_service.analyze_sentiment(request.text)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/analyze-document", response_model=Dict[str, Any])
async def analyze_document_sentiment(request: HTMLRequest):
    """
    Analyze sentiment for each sentence in HTML document
    
    Returns highlighted HTML with sentiment colors and statistics
    """
    try:
        if not request.html_content.strip():
            raise HTTPException(status_code=400, detail="HTML content cannot be empty")
        
        # Analyze document sentiment
        result = finbert_service.analyze_document_sentiment(request.html_content)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.post("/analyze-text-sentences", response_model=Dict[str, Any])
async def analyze_text_sentences(request: TextRequest):
    """
    Analyze sentiment for each sentence in plain text
    
    Returns sentence-by-sentence analysis with statistics
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Analyze text sentiment
        result = finbert_service.analyze_text_sentiment(request.text)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@router.get("/model-status")
async def get_model_status():
    """
    Check FinBERT model status and availability
    """
    try:
        is_available = finbert_service.is_model_available()
        
        return {
            "model_loaded": is_available,
            "model_name": Config.FINBERT_MODEL,
            "transformers_available": True,  # Since service initialized
            "huggingface_api_key_configured": bool(Config.HUGGINGFACE_API_KEY),
            "recommendations": [
                "Set HUGGINGFACE_API_KEY environment variable" if not Config.HUGGINGFACE_API_KEY else None,
                "Load model using /load-model endpoint" if not is_available else None
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error checking model status: {str(e)}")

@router.post("/load-model")
async def load_finbert_model(model_name: Optional[str] = None):
    """
    Load FinBERT model for sentiment analysis
    """
    try:
        success = finbert_service.load_model(model_name)
        
        if success:
            return {
                "success": True,
                "message": "FinBERT model loaded successfully",
                "model_name": model_name or Config.FINBERT_MODEL
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to load FinBERT model")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@router.get("/sentiment-colors")
async def get_sentiment_colors():
    """
    Get sentiment color mapping used for highlighting
    """
    return {
        "colors": finbert_service.get_sentiment_colors(),
        "labels": {
            "very positive": "Strong positive sentiment",
            "positive": "Moderate positive sentiment", 
            "neutral": "Neutral sentiment",
            "negative": "Moderate negative sentiment",
            "very negative": "Strong negative sentiment"
        }
    }

@router.get("/sentiment-labels")
async def get_sentiment_labels():
    """
    Get information about sentiment labels used by FinBERT
    """
    return {
        "standard_labels": ["positive", "negative", "neutral"],
        "extended_labels": ["very positive", "positive", "neutral", "negative", "very negative"],
        "model_output": "FinBERT typically outputs: positive, negative, neutral",
        "scoring": "Confidence score ranges from 0.0 to 1.0",
        "usage": "Higher score indicates higher confidence in the predicted label"
    }
