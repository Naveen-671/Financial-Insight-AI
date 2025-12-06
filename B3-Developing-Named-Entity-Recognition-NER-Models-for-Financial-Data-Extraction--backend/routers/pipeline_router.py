from fastapi import APIRouter, UploadFile, File, HTTPException, Form
import base64
import io
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import os
import json

from services.docling_service import DoclingService
from services.ner_service import NERService
from services.langextract_service import LangExtractService
from services.finbert_service import FinBERTService
from config import Config

router = APIRouter()

# DoclingService imported from services

# Initialize services
docling_service = DoclingService()
ner_service = NERService()
langextract_service = LangExtractService()
finbert_service = FinBERTService()


class PipelineOptions(BaseModel):
    include_ner: bool = True
    include_langextract: bool = True
    include_finbert: bool = True
    ner_model_path: Optional[str] = None
    langextract_models: Optional[List[str]] = None
    langextract_api_key: Optional[str] = None
    clauses_prompt: Optional[str] = None
    finbert_model_name: Optional[str] = None


class TextPipelineRequest(BaseModel):
    text: str
    options: PipelineOptions = PipelineOptions()


@router.get("/pipeline-status")
async def get_pipeline_status():
    return {
        "ner": {
            "available": ner_service.is_model_available(),
            "model_path": str(Config.NER_MODEL_PATH),
            "model_exists": Config.NER_MODEL_PATH.exists()
        }
    }

@router.post("/test-upload")
async def test_upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        return {
            "success": True,
            "filename": file.filename,
            "content_type": file.content_type,
            "size": len(content),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Optional model (currently not used directly in the endpoint, but kept for future use)
class ProcessDocumentRequest(BaseModel):
    include_ner: bool = True
    include_langextract: bool = True
    include_finbert: bool = True
    ner_model_path: Optional[str] = None
    langextract_models: Optional[str] = None
    langextract_api_key: Optional[str] = None
    finbert_model_name: Optional[str] = None


@router.post("/process-document", response_model=Dict[str, Any])
async def process_document_pipeline(
    file: UploadFile = File(...),
    include_ner: bool = Form(True),
    include_langextract: bool = Form(True),
    include_finbert: bool = Form(True),
    ner_model_path: Optional[str] = Form(None),
    langextract_models: Optional[str] = Form(None),
    langextract_api_key: Optional[str] = Form(None),
    clauses_prompt: Optional[str] = Form(None),
    finbert_model_name: Optional[str] = Form(None),
):
    """
    End-to-end document processing pipeline.
    Expects multipart/form-data with:
    - file (UploadFile)
    - include_ner, include_langextract, include_finbert (bool, optional)
    - ner_model_path, langextract_models, langextract_api_key, finbert_model_name (optional)
    """
    file_path: Optional[str] = None

    try:
        # Read file content
        file_content = await file.read()
        await file.seek(0)

        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Validate file size and type
        validation_result = docling_service.validate_file(file.filename, len(file_content))
        if not validation_result.get("valid", False):
            raise HTTPException(status_code=400, detail=validation_result.get("error", "Invalid file"))

        # Save uploaded file to temp/storage
        try:
            file_path = docling_service.save_uploaded_file(file_content, file.filename)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

        # Process the file with Docling
        try:
            docling_result = docling_service.extract_text_from_file(file_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Docling processing failed: {str(e)}")

        if not docling_result.get("success", False):
            error_msg = docling_result.get("error", "Unknown error during text extraction")
            raise HTTPException(status_code=500, detail=f"Text extraction failed: {error_msg}")

        extracted_text: str = docling_result.get("text", "") or ""
        extracted_html: str = docling_result.get("html", "") or ""

        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the document",
            )

        # Initialize pipeline results
        pipeline_result: Dict[str, Any] = {
            "success": True,
            "file_info": {
                "original_filename": file.filename,
                "file_size": len(file_content),
                "file_type": docling_result.get("metadata", {}).get("file_type", "unknown"),
                "extraction_method": docling_result.get("metadata", {}).get("extraction_method", "unknown"),
            },
            "text_extraction": {
                "success": True,
                "text_length": len(extracted_text),
                "html_available": bool(extracted_html),
                "metadata": docling_result.get("metadata", {}),
            },
            "pipeline_steps": [],
        }

        # Helper: parse langextract_models string into list if provided
        parsed_langextract_models: Optional[List[str]] = None
        if langextract_models:
            try:
                # Try to parse as JSON list
                parsed = json.loads(langextract_models)
                if isinstance(parsed, list):
                    parsed_langextract_models = parsed
                elif isinstance(parsed, str):
                    # maybe it's a single model name packed in a JSON string
                    parsed_langextract_models = [parsed]
            except json.JSONDecodeError:
                # Fallback: comma-separated string
                parsed_langextract_models = [m.strip() for m in langextract_models.split(",") if m.strip()]

        # Step 1: NER (if requested)
        if include_ner:
            try:
                if ner_model_path:
                    ner_service.load_model(ner_model_path)

                ner_result = ner_service.extract_entities(extracted_text)
                pipeline_result["ner"] = ner_result
                pipeline_result["pipeline_steps"].append("ner")
            except Exception as e:
                pipeline_result["ner"] = {"success": False, "error": str(e)}

        # Step 2: LangExtract (if requested)
        if include_langextract:
            try:
                # Generate a document ID based on filename hash or random
                import hashlib
                doc_id_hash = hashlib.md5(file.filename.encode()).hexdigest()[:8]
                document_id = f"doc_{doc_id_hash}"

                langextract_result = langextract_service.extract_clauses(
                    text=extracted_text,
                    models=parsed_langextract_models,
                    api_key=langextract_api_key,
                    document_id=document_id,
                    user_instruction=clauses_prompt
                )
                pipeline_result["langextract"] = langextract_result
                pipeline_result["pipeline_steps"].append("langextract")
            except Exception as e:
                pipeline_result["langextract"] = {"success": False, "error": str(e)}

        # Step 3: FinBERT (if requested)
        if include_finbert:
            try:
                if finbert_model_name:
                    finbert_service.load_model(finbert_model_name)

                if extracted_html:
                    finbert_result = finbert_service.analyze_document_sentiment(extracted_html)
                else:
                    finbert_result = finbert_service.analyze_text_sentiment(extracted_text)

                pipeline_result["finbert"] = finbert_result
                pipeline_result["pipeline_steps"].append("finbert")
            except Exception as e:
                pipeline_result["finbert"] = {"success": False, "error": str(e)}

        # Add summary
        pipeline_result["summary"] = {
            "total_steps_completed": len(pipeline_result["pipeline_steps"]),
            "steps_completed": pipeline_result["pipeline_steps"],
            "text_length": len(extracted_text),
            "processing_success": all(
                step_result.get("success", False)
                for step_name, step_result in pipeline_result.items()
                if step_name in ["ner", "langextract", "finbert"]
            ),
        }

        return JSONResponse(content=pipeline_result)

    except HTTPException:
        # Re-raise HTTPException unchanged
        raise
    except Exception as e:
        # Catch-all unexpected error
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error during pipeline processing: {str(e)}",
        )
    finally:
        # Cleanup temp file if it exists
        if file_path and os.path.exists(file_path):
            try:
                docling_service.cleanup_file(file_path)
            except Exception as e:
                # Log cleanup issue but don't break the request
                print(f"Warning: Failed to clean up file {file_path}: {str(e)}")


@router.post("/process-text", response_model=Dict[str, Any])
async def process_text_pipeline(request: TextPipelineRequest):
    """
    End-to-end text processing pipeline.

    Processes pre-extracted text through NER, LangExtract, and FinBERT.
    """
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        pipeline_result: Dict[str, Any] = {
            "success": True,
            "text_info": {
                "text_length": len(request.text),
                "word_count": len(request.text.split()),
            },
            "pipeline_steps": [],
        }

        # NER
        if request.options.include_ner:
            try:
                if request.options.ner_model_path:
                    ner_service.load_model(request.options.ner_model_path)

                ner_result = ner_service.extract_entities(request.text)
                pipeline_result["ner"] = ner_result
                pipeline_result["pipeline_steps"].append("ner")
            except Exception as e:
                pipeline_result["ner"] = {"success": False, "error": str(e)}

        # LangExtract
        if request.options.include_langextract:
            try:
                langextract_result = langextract_service.extract_clauses(
                    text=request.text,
                    models=request.options.langextract_models,
                    api_key=request.options.langextract_api_key,
                    user_instruction=request.options.clauses_prompt
                )
                pipeline_result["langextract"] = langextract_result
                pipeline_result["pipeline_steps"].append("langextract")
            except Exception as e:
                pipeline_result["langextract"] = {"success": False, "error": str(e)}

        # FinBERT
        if request.options.include_finbert:
            try:
                if request.options.finbert_model_name:
                    finbert_service.load_model(request.options.finbert_model_name)

                finbert_result = finbert_service.analyze_text_sentiment(request.text)
                pipeline_result["finbert"] = finbert_result
                pipeline_result["pipeline_steps"].append("finbert")
            except Exception as e:
                pipeline_result["finbert"] = {"success": False, "error": str(e)}

        # Summary
        pipeline_result["summary"] = {
            "total_steps_completed": len(pipeline_result["pipeline_steps"]),
            "steps_completed": pipeline_result["pipeline_steps"],
            "processing_success": all(
                step_result.get("success", False)
                for step_name, step_result in pipeline_result.items()
                if step_name in ["ner", "langextract", "finbert"]
            ),
        }

        return pipeline_result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text pipeline processing failed: {str(e)}",
        )


@router.get("/pipeline-status")
async def get_pipeline_status():
    """
    Check status of all pipeline components.
    """
    try:
        return {
            "docling": {
                "available": True,
                "supported_formats": list(Config.ALLOWED_EXTENSIONS),
            },
            "ner": {
                "available": ner_service.is_model_available(),
                "model_path": str(Config.NER_MODEL_PATH),
                "model_exists": Config.NER_MODEL_PATH.exists(),
            },
            "langextract": {
                "available": langextract_service.is_available(),
                "api_key_configured": bool(Config.GOOGLE_API_KEY),
                "supported_models": Config.LANGEXTRACT_MODELS,
            },
            "finbert": {
                "available": finbert_service.is_model_available(),
                "model_name": Config.FINBERT_MODEL,
                "api_key_required": False,
            },
            "overall_status": {
                "ready_for_document_processing": True,
                "ready_for_full_pipeline": all(
                    [
                        ner_service.is_model_available(),
                        langextract_service.is_available(),
                        finbert_service.is_model_available(),
                    ]
                ),
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking pipeline status: {str(e)}",
        )


@router.get("/pipeline-options")
async def get_pipeline_options():
    """
    Get available pipeline options and configurations.
    """
    return {
        "processing_steps": {
            "ner": {
                "description": "Named Entity Recognition",
                "outputs": ["entities", "visualization_html"],
                "model_options": {
                    "default": str(Config.NER_MODEL_PATH),
                    "custom": "Custom model path",
                },
            },
            "langextract": {
                "description": "Financial/Legal Clause Extraction",
                "outputs": ["extracted_clauses", "visualization_html", "json_results"],
                "model_options": Config.LANGEXTRACT_MODELS,
                "requires_api_key": True,
            },
            "finbert": {
                "description": "Sentiment Analysis",
                "outputs": ["sentiment_results", "highlighted_html", "statistics"],
                "model_options": {
                    "default": Config.FINBERT_MODEL,
                    "custom": "Custom model name",
                },
            },
        },
        "file_formats": {
            "supported": list(Config.ALLOWED_EXTENSIONS),
            "max_size_mb": Config.MAX_FILE_SIZE / (1024 * 1024),
        },
        "output_types": {
            "json": "Structured results in JSON format",
            "html": "Interactive visualizations",
            "statistics": "Summary statistics and metrics",
        },
    }
