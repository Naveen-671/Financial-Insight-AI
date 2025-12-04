# Financial Insight - AI-Powered Document Analysis Platform

Financial Insight is a comprehensive platform designed to extract actionable intelligence from financial and legal documents. It leverages advanced AI models for Named Entity Recognition (NER), Sentiment Analysis, Clause Extraction, and Contextual Question Answering (RAG).

## 🚀 Key Features

### 1. Named Entity Recognition (NER)
- **Automatic Extraction:** Identifies key entities such as Organizations, Dates, Money, Locations, and more.
- **Visualization:** Interactive highlighting of entities within the document text.
- **Analytics:** Entity distribution charts and frequency analysis.
- **Powered By:** spaCy (`en_core_web_sm`) and custom-trained models.

### 2. Sentiment Analysis
- **Financial Context:** specifically tuned for financial texts using FinBERT.
- **Granular Analysis:** Sentence-level sentiment scoring (Positive, Negative, Neutral).
- **Visual Heatmap:** Color-coded document view to instantly spot sentiment trends.
- **Powered By:** `ProsusAI/finbert` via Hugging Face Transformers.

### 3. Clause Extraction
- **Legal Intelligence:** Automatically extracts and categorizes critical legal clauses (e.g., Liability, Termination, Payment).
- **Risk Profiling:** Categorizes clauses by risk level (High, Medium, Low).
- **Powered By:** Groq (`llama-3.1-8b-instant`) for high-speed, accurate extraction.

### 4. RAG Chatbot (Contextual AI Guide)
- **Interactive Q&A:** Chat with your documents to get precise answers.
- **Context Aware:** Uses Retrieval-Augmented Generation to ground answers in document content.
- **Powered By:** 
    - **LLM:** Groq (`llama-3.1-8b-instant`)
    - **Embeddings:** Google Gemini (`text-embedding-004`)
    - **Vector DB:** Pinecone (with in-memory fallback)

---

## 🛠️ Tech Stack

### Backend
- **Framework:** FastAPI (Python)
- **AI/ML:** PyTorch, Transformers, spaCy, LangChain
- **LLM Providers:** Groq, Google Gemini
- **Vector Database:** Pinecone
- **Document Processing:** PyPDF, python-docx

### Frontend
- **Framework:** React + Vite
- **Styling:** Tailwind CSS
- **UI Components:** Radix UI, Lucide React
- **Visualization:** Recharts, Framer Motion

---

## 📋 Prerequisites

- **Python** 3.9+
- **Node.js** 16+
- **API Keys:**
    - `GROQ_API_KEY` (for RAG and Clause Extraction)
    - `GOOGLE_API_KEY` (for Embeddings)
    - `PINECONE_API_KEY` (Optional, for production vector storage)

---

## ⚡ Installation & Setup

### 1. Backend Setup

Navigate to the backend directory:
```bash
cd "B3-Developing-Named-Entity-Recognition-NER-Models-for-Financial-Data-Extraction--backend"
```

Create a virtual environment (recommended):
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

**Configuration:**
Create a `.env` file in the backend directory with your API keys:
```env
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
PINECONE_API_KEY=your_pinecone_key (optional)
```

Run the server:
```bash
python app.py
```
*The backend will start on `http://localhost:8001`*

### 2. Frontend Setup

Navigate to the frontend directory:
```bash
cd "B3-Developing-Named-Entity-Recognition-NER-Models-for-Financial-Data-Extraction--Financial_Insight"
```

Install dependencies:
```bash
npm install
```

Run the development server:
```bash
npm run dev
```
*The frontend will start on `http://localhost:5173` (typically)*

---

## 📖 Usage Guide

1.  **Upload:** Go to the "Upload" page and select a PDF, DOCX, or TXT file.
2.  **Analyze:** The system will automatically process the document.
3.  **View Results:**
    -   **Dashboard:** See high-level metrics.
    -   **NER:** Explore extracted entities.
    -   **Sentiment:** View sentiment distribution and heatmap.
    -   **Clauses:** Review extracted legal clauses and risk profile.
4.  **Chat:** Switch to the "RAG" tab to ask specific questions about the uploaded document.

---

## 📄 API Documentation

Full API documentation is available when the backend is running at:
-   **Swagger UI:** `http://localhost:8001/docs`
-   **ReDoc:** `http://localhost:8001/redoc`

---

*Built with ❤️ for Financial Insight*
