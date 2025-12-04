# Financial Data Extraction and Analysis API

A comprehensive FastAPI backend for extracting, analyzing, and querying financial documents using state-of-the-art NLP and AI models.

## Features

- **Docling**: Extract raw text from HTML, PDF, DOCX, and other document formats
- **NER (Named Entity Recognition)**: Identify entities like company, person, money, etc. using trained spaCy models
- **LangExtract**: Extract financial and legal clauses using AI models (Gemini)
- **FinBERT**: Analyze sentiment of financial text using FinBERT model
- **RAG (Retrieval-Augmented Generation)**: Advanced document querying with contextual understanding
- **Pipeline**: End-to-end processing combining all modules

## Quick Start

### Prerequisites

- Python 3.11+
- Google API Key (for LangExtract and RAG)
- Hugging Face API Key (for FinBERT)
- Trained spaCy NER model
- Additional Python packages (installed via requirements.txt)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd backend_Checked_1
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Place your trained NER model:
```bash
# Copy your trained spaCy model to:
models/ner_model/model-best
```

### Running the Application

```bash
python app.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### Health Check
- `GET /health` - Check API status
- `GET /` - Root endpoint

### Docling (Text Extraction)
- `POST /api/v1/docling/extract-text` - Extract text from uploaded file
- `GET /api/v1/docling/supported-formats` - Get supported file formats

### NER (Named Entity Recognition)
- `POST /api/v1/ner/extract-entities` - Extract entities from text
- `GET /api/v1/ner/model-status` - Check NER model status
- `POST /api/v1/ner/load-model` - Load NER model
- `POST /api/v1/ner/train-model` - Train new NER model

### LangExtract (Clause Extraction)
- `POST /api/v1/langextract/extract-clauses` - Extract financial/legal clauses
- `GET /api/v1/langextract/service-status` - Check LangExtract service status
- `PUT /api/v1/langextract/update-prompt` - Update extraction prompt
- `POST /api/v1/langextract/add-example` - Add training example

### RAG (Retrieval-Augmented Generation)
- `POST /api/v1/rag/query` - Query documents with natural language questions
- `POST /api/v1/rag/process-document` - Process and index a document for querying
- `GET /api/v1/rag/status` - Check RAG service status

### FinBERT (Sentiment Analysis)
- `POST /api/v1/finbert/analyze-sentiment` - Analyze sentiment of text
- `POST /api/v1/finbert/analyze-document` - Analyze sentiment in HTML document
- `POST /api/v1/finbert/analyze-text-sentences` - Analyze sentiment sentence by sentence
- `GET /api/v1/finbert/model-status` - Check FinBERT model status

### Pipeline (End-to-End Processing)
- `POST /api/v1/pipeline/process-document` - Process uploaded document through all modules
- `POST /api/v1/pipeline/process-text` - Process text through all modules
- `GET /api/v1/pipeline/pipeline-status` - Check status of all pipeline components
- `GET /api/v1/pipeline/pipeline-options` - Get available pipeline options

## Environment Variables

```bash
# Required for LangExtract and RAG
GOOGLE_API_KEY=your_google_api_key_here

# Required for FinBERT
HUGGINGFACE_API_KEY=your_huggingface_api_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000

# Model Paths
NER_MODEL_PATH=./models/ner_model/model-best

# File Storage
UPLOAD_DIR=./uploads
OUTPUT_DIR=./outputs

# RAG Configuration
RAG_MODEL_NAME=google_genai:gemini-2.5-flash-lite
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## File Structure

```
backend/
│
├── .env.example         # Example environment variables
├── .gitignore           # Git ignore file
├── API_DOCUMENTATION.md # API documentation
├── README.md            # Project documentation
├── __init__.py          # Python package initializer
├── app.py               # Main FastAPI application
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
├── test_server.py       # Test server script
│
├── models/              # ML models directory
│   └── ner_model/       # NER model files
│
├── outputs/             # Directory for output files
│
├── routers/             # API route definitions
│   ├── __init__.py
│   ├── docling_router.py     # Document processing routes
│   ├── finbert_router.py     # Sentiment analysis routes
│   ├── langextract_router.py # Legal/financial clause extraction
│   ├── ner_router.py         # Named Entity Recognition routes
│   ├── pipeline_router.py    # End-to-end processing pipeline
│   └── rag_router.py         # RAG system routes
│
├── services/            # Business logic and service layer
│   ├── __init__.py
│   ├── docling_service.py     # Document processing service
│   ├── finbert_service.py     # Sentiment analysis service
│   ├── langextract_service.py # Clause extraction service
│   ├── ner_service.py         # NER processing service
│   └── rag_service.py         # RAG system service
│
├── uploads/             # Directory for uploaded files
│
└── venv/                # Python virtual environment (created during setup)
```

## Usage Examples

### Extract Text from Document
```bash
curl -X POST "http://localhost:8000/api/v1/docling/extract-text" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### Extract Entities
```bash
curl -X POST "http://localhost:8000/api/v1/ner/extract-entities" \
  -H "Content-Type: application/json" \
  -d '{"text": "Market analysts at MorganEast Research forecast growth."}'
```

### Full Pipeline Processing
```bash
# Process document through all modules
curl -X POST "http://localhost:8000/api/v1/pipeline/process-document" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@financial_document.pdf" \
  -F 'options={"include_ner": true, "include_langextract": true, "include_finbert": true}'

# Query processed documents using RAG
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key financial risks mentioned?"}'

# Process and index a new document for RAG
curl -X POST "http://localhost:8000/api/v1/rag/process-document" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@financial_report.pdf"
```

## Output Files

- **NER**: HTML visualization with highlighted entities
- **LangExtract**: JSON results and HTML visualization
- **FinBERT**: HTML with sentiment-highlighted sentences
- **RAG**: Processed document chunks and query results
- **Pipeline**: Combined results with all visualizations

All output files are saved in the `outputs/` directory and accessible via `/outputs/{filename}`.

## Development

### Running Tests
```bash
pytest
```

### Code Style
This project uses standard Python formatting. Ensure your code follows PEP 8 guidelines.

## Troubleshooting

### Common Issues

1. **NER Model Not Found**: Ensure your trained spaCy model is placed at `models/ner_model/model-best`
2. **RAG Service Errors**: 
   - Ensure Google API key has access to the Gemini model
   - Check that documents are properly processed before querying
   - Verify that the document text is being extracted correctly
3. **Missing Dependencies**: If you encounter module errors, run:
   ```bash
   pip install -r requirements.txt
   pip install langchain langchain-google-genai pypdf langchain-community
   ```
2. **API Key Errors**: Set `GOOGLE_API_KEY` in your environment
3. **Memory Issues**: For large documents, consider increasing system RAM or using smaller chunks
4. **LangExtract Not Available**: Install with `pip install langextract`

### Logs

Check the application logs for detailed error information. Logs are printed to console by default.

## License

This project is licensed under the MIT License.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

For issues and questions, please create an issue in the repository.
