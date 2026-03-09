import os
from pathlib import Path
from typing import Dict, Any

class Config:
    # Base paths
    BASE_DIR = Path(__file__).parent
    UPLOADS_DIR = BASE_DIR / "uploads"
    OUTPUTS_DIR = BASE_DIR / "outputs"
    MODELS_DIR = BASE_DIR / "models"
    
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Model settings (API-based, no local models)
    NER_MODEL_NAME = "dslim/bert-base-NER"  # HuggingFace model for NER
    NER_MODEL_PATH = MODELS_DIR / "ner_model" / "model-best"  # Legacy path (unused in API mode)
    
    # API Keys (set from environment)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # LangExtract settings
    LANGEXTRACT_MODELS = ["gemini-2.5-flash","gemini-2.5-flash-lite", "gemini-2.5-pro"]
    
    # FinBERT settings (API-based)
    FINBERT_MODEL = "ProsusAI/finbert"
    
    # File size limits
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS = {".pdf", ".txt", ".html", ".docx", ".doc"}
    
    # Sentiment colors
    SENTIMENT_COLORS = {
        "very positive": "#bbf7d0",  # Light Green
        "positive": "#dcfce7",       # Very Light Green
        "neutral": "#f3f4f6",        # Very Light Gray
        "negative": "#fee2e2",       # Very Light Red
        "very negative": "#fecaca"   # Light Red
    }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        cls.UPLOADS_DIR.mkdir(exist_ok=True)
        cls.OUTPUTS_DIR.mkdir(exist_ok=True)
        cls.MODELS_DIR.mkdir(exist_ok=True)
        
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        issues = []
        
        if not cls.GOOGLE_API_KEY:
            issues.append("GOOGLE_API_KEY not set")
        
        if not cls.HUGGINGFACE_API_KEY:
            issues.append("HUGGINGFACE_API_KEY not set (NER and FinBERT will use fallback)")
            
        if not cls.GROQ_API_KEY:
            issues.append("GROQ_API_KEY not set (RAG and LangExtract will not work)")
            
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
