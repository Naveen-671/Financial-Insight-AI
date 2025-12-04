# In app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import routers
from routers import (
    # docling_router, 
    ner_router, 
    langextract_router, 
    finbert_router, 
    pipeline_router,
    rag_router,
    export_router  # Add export router
)

# Create FastAPI app
app = FastAPI(title="Document Processing API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rag_router.router)  # Prefix defined in router file
app.include_router(export_router.router)  # Add export router
app.include_router(pipeline_router.router, prefix="/api/v1/pipeline", tags=["Pipeline"])
app.include_router(ner_router.router, prefix="/api/v1/ner", tags=["NER"])
app.include_router(langextract_router.router, prefix="/api/v1/langextract", tags=["LangExtract"])
app.include_router(finbert_router.router, prefix="/api/v1/finbert", tags=["FinBERT"])
# app.include_router(docling_router.router)

# Create necessary directories
os.makedirs("uploads", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

# Health check endpoint
@app.get("/")
async def root():
    return {"status": "ok", "message": "Document Processing API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8001, reload=True)