# Backend API Documentation

## Table of Contents
- [Document Processing](#document-processing)
- [Sentiment Analysis](#sentiment-analysis)
- [Legal & Financial Clause Extraction](#legal--financial-clause-extraction)
- [Named Entity Recognition](#named-entity-recognition-ner)
- [RAG (Retrieval-Augmented Generation)](#retrieval-augmented-generation-rag)
- [End-to-End Pipeline](#end-to-end-pipeline)

## Document Processing

### Extract Text from Document
- **Endpoint**: `POST /extract-text`
- **Description**: Extract text from uploaded documents (PDF, TXT, HTML, DOCX, DOC)
- **Request**: `multipart/form-data` with file
- **Response**: 
  ```json
  {
    "success": bool,
    "text": "extracted text",
    "html": "formatted HTML (if available)",
    "metadata": {
      "file_type": "pdf",
      "page_count": 1,
      "extraction_method": "pypdf"
    }
  }
  ```

### Get Supported Formats
- **Endpoint**: `GET /supported-formats`
- **Description**: List supported file formats and size limits
- **Response**:
  ```json
  {
    "supported_extensions": ["pdf", "txt", "docx", "doc", "html"],
    "max_file_size_mb": 10
  }
  ```

## Sentiment Analysis

### Analyze Text Sentiment
- **Endpoint**: `POST /analyze-sentiment`
- **Description**: Analyze sentiment of financial text
- **Request**: 
  ```json
  {
    "text": "Company X reported record profits this quarter.",
    "model_name": "optional-model-name"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "sentiment": "positive",
    "score": 0.95,
    "sentences": [
      {
        "text": "Company X reported record profits this quarter.",
        "sentiment": "positive",
        "score": 0.95
      }
    ]
  }
  ```

### Get Sentiment Colors
- **Endpoint**: `GET /sentiment-colors`
- **Description**: Get color mapping for sentiment visualization
- **Response**:
  ```json
  {
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral": "#9E9E9E"
  }
  ```

## Legal & Financial Clause Extraction

### Extract Clauses
- **Endpoint**: `POST /extract-clauses`
- **Description**: Extract legal and financial clauses from text
- **Request**:
  ```json
  {
    "text": "The interest rate shall be 5% per annum...",
    "models": ["gemini-2.5-flash"],
    "api_key": "optional-api-key"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "clauses": [
      {
        "type": "interest_clause",
        "text": "The interest rate shall be 5% per annum",
        "attributes": {
          "interest_rate": "5%",
          "period": "per annum"
        }
      }
    ]
  }
  ```

### Get Supported Clause Types
- **Endpoint**: `GET /clause-types`
- **Description**: List supported clause types and their attributes

## Named Entity Recognition (NER)

### Extract Entities
- **Endpoint**: `POST /extract-entities`
- **Description**: Extract named entities from text
- **Request**:
  ```json
  {
    "text": "Apple Inc. reported $1B revenue in Q1 2023.",
    "model_path": "optional/path/to/model"
  }
  ```
- **Response**:
  ```json
  {
    "success": true,
    "entities": [
      {
        "text": "Apple Inc.",
        "label": "ORG",
        "start_char": 0,
        "end_char": 9
      },
      {
        "text": "$1B",
        "label": "MONEY",
        "start_char": 18,
        "end_char": 20
      },
      {
        "text": "Q1 2023",
        "label": "DATE",
        "start_char": 33,
        "end_char": 39
      }
    ]
  }
  ```

## Retrieval-Augmented Generation (RAG)

### Query RAG System
- **Endpoint**: `POST /api/v1/rag/query`
- **Description**: Get AI-generated answers based on document content
- **Request**:
  ```json
  {
    "question": "What is the interest rate?",
    "document_path": "documents/contract.pdf"
  }
  ```
- **Response**:
  ```json
  {
    "answer": "The interest rate is 5% per annum.",
    "sources": ["documents/contract.pdf"],
    "confidence": 0.92
  }
  ```

## End-to-End Pipeline

### Process Document (All-in-One)
- **Endpoint**: `POST /process-document`
- **Description**: Process document through all available services
- **Request**: `multipart/form-data`
  - `file`: Document file
  - `include_ner`: boolean (default: true)
  - `include_langextract`: boolean (default: true)
  - `include_finbert`: boolean (default: true)
  - `ner_model_path`: string (optional)
  - `langextract_models`: string (comma-separated)
  - `finbert_model_name`: string (optional)
- **Response**: Combined results from all enabled services

### Get Pipeline Status
- **Endpoint**: `GET /pipeline-status`
- **Description**: Check status of all pipeline components
- **Response**:
  ```json
  {
    "ner": {
      "available": true,
      "model_path": "/path/to/ner/model"
    },
    "langextract": {
      "available": true,
      "models": ["gemini-2.5-flash"]
    },
    "finbert": {
      "available": true,
      "model_name": "yiyanghkust/finbert-tone"
    }
  }
  ```

## Error Handling
All endpoints return standardized error responses:
```json
{
  "detail": "Error message"
}
```

Common HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid input)
- `404`: Resource not found
- `500`: Internal server error

## Authentication
Currently, the API does not require authentication. However, some features may require API keys to be set in environment variables:
- `GOOGLE_API_KEY`: For LangExtract and RAG services

## Rate Limiting
No rate limiting is currently implemented. Please be considerate with request volumes during development.
