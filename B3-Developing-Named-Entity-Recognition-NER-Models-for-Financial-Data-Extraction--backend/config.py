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
    # Model paths
    NER_MODEL_NAME = "en_core_web_sm"  # Default spaCy model to use
    NER_MODEL_PATH = MODELS_DIR / "ner_model" / "model-best"  # Custom model path (if available)
    
    # Ensure the model path exists
    if not NER_MODEL_PATH.exists():
        print(f"Warning: NER model not found at {NER_MODEL_PATH}")
        # Try to create the directory structure if it doesn't exist
        NER_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    
    # API Keys (set from environment)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    # LangExtract settings
    LANGEXTRACT_MODELS = ["gemini-2.5-flash","gemini-2.5-flash-lite", "gemini-2.5-pro"]
    
    # FinBERT settings
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
            
        if not cls.NER_MODEL_PATH.exists():
            issues.append(f"NER model not found at {cls.NER_MODEL_PATH}")
            
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
