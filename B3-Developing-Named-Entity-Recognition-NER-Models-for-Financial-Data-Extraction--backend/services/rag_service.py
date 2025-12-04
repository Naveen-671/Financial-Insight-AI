"""
RAG (Retrieval-Augmented Generation) Service Module

This module provides document processing and question-answering capabilities
using Google's Gemini model, Gemini embeddings, and Pinecone vector store.
"""

import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import requests
import numpy as np
from pydantic import BaseModel, Field
from pypdf import PdfReader
import hashlib
import time

# Pinecone import
try:
    from pinecone.grpc import PineconeGRPC as Pinecone
    from pinecone import ServerlessSpec
    HAS_PINECONE = True
except ImportError:
    try:
        from pinecone import Pinecone, ServerlessSpec
        HAS_PINECONE = True
    except ImportError:
        HAS_PINECONE = False
        Pinecone = None
        ServerlessSpec = None
        print("Warning: Pinecone not installed. Using in-memory storage only.")

# Groq import
try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False
    print("Warning: Groq not installed.")

# Text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    HAS_TEXT_SPLITTER = True
except ImportError:
    HAS_TEXT_SPLITTER = False

# DOCX support
try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

class RAGRequest(BaseModel):
    """Request model for RAG queries."""
    question: str
    document_path: Optional[str] = None
    chunk_size: int = 1000
    chunk_overlap: int = 200

class RAGResponse(BaseModel):
    """Response model for RAG queries."""
    success: bool
    answer: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class RAGService:
    """Service for handling RAG using Gemini and Pinecone."""
    
    def __init__(self):
        from config import Config
        self.api_key = None # Google API Key for embeddings
        self.groq_api_key = None # Groq API Key for generation
        self.pinecone_api_key = None
        self.pinecone_client = None
        self.groq_client = None
        self.index = None
        self.index_name = "financial-docs"
        self.initialized = False
        self.documents = []  # Fallback in-memory storage

    def initialize(self, api_key: Optional[str] = None, pinecone_api_key: Optional[str] = None, groq_api_key: Optional[str] = None):
        """Initialize the RAG service with API keys."""
        try:
            from config import Config
            
            self.api_key = api_key or Config.GOOGLE_API_KEY
            self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
            self.groq_api_key = groq_api_key or Config.GROQ_API_KEY
            
            if not self.api_key:
                print("Warning: Google API key not configured for RAG (Embeddings)")
            
            if self.groq_api_key and HAS_GROQ:
                try:
                    self.groq_client = Groq(api_key=self.groq_api_key)
                    print("Groq client initialized.")
                except Exception as e:
                    print(f"Failed to initialize Groq client: {e}")
            else:
                print("Warning: Groq API key not found or Groq not installed. Generation might fail.")

            # Mark as initialized even without Pinecone
            # Pinecone will initialize on first document upload
            self.initialized = True
            print(f"RAG service initialized. Pinecone: {bool(self.pinecone_api_key)}, Google API: {bool(self.api_key)}, Groq: {bool(self.groq_client)}")
            return True
            
        except Exception as e:
            print(f"RAG initialization warning: {e}")
            self.initialized = True  # Allow to continue with in-memory
            return True

    def _ensure_pinecone_index(self):
        """Initialize Pinecone index on first use."""
        if self.index is not None or not HAS_PINECONE or not self.pinecone_api_key:
            return
            
        try:
            self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
            
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pinecone_client.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"Creating Pinecone index: {self.index_name}")
                try:
                    self.pinecone_client.create_index(
                        name=self.index_name,
                        dimension=768,  # Gemini embedding dimension
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
                    # Wait for index to be ready
                    time.sleep(10)
                except Exception as e:
                    print(f"Failed to create index (might already exist or permission error): {e}")
            
            # Use the specific host provided by the user to ensure connection
            # Host: https://financial-insight-cgn8neb.svc.aped-4627-b74a.pinecone.io
            self.index = self.pinecone_client.Index(
                name=self.index_name, 
                host="https://financial-insight-cgn8neb.svc.aped-4627-b74a.pinecone.io"
            )
            print(f"Connected to Pinecone index: {self.index_name} at specific host")
        except Exception as e:
            print(f"Pinecone initialization failed: {e}. Using in-memory storage.")
            self.index = None

    def _generate_content(self, prompt: str) -> str:
        """Generate content using Groq API."""
        if not self.groq_client:
            return "Error: Groq client not initialized. Please check GROQ_API_KEY in .env or config."

        try:
            print("Sending request to Groq (llama3-8b-8192)...", flush=True)
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="llama-3.1-8b-instant",
                temperature=0.7,
                max_tokens=2048,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            print(f"Groq generation error: {e}", flush=True)
            return f"Error generating content: {str(e)}"

    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text using Gemini Embedding API with retry logic."""
        if not self.api_key:
            return None
            
        url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.api_key}"
        payload = {
            "model": "models/text-embedding-004",
            "content": {"parts": [{"text": text}]}
        }
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(url, json=payload, timeout=30)
                
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"Embedding rate limit hit (429). Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                result = response.json()
                return result["embedding"]["values"]
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Error getting embedding after retries: {e}")
                    return None
                time.sleep(1)
                
        return None

    def _extract_text_from_file(self, file_path: Union[str, Path]) -> str:
        """Extract text from PDF, DOCX, or TXT file."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        if ext == ".pdf":
            reader = PdfReader(str(file_path))
            return "\n".join(page.extract_text() for page in reader.pages)
        elif ext == ".txt":
            return file_path.read_text(encoding="utf-8")
        elif ext in [".docx", ".doc"] and HAS_DOCX:
            doc = DocxDocument(str(file_path))
            return "\n".join(paragraph.text for paragraph in doc.paragraphs)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def _split_text(self, text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into chunks."""
        if HAS_TEXT_SPLITTER:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            return splitter.split_text(text)
        else:
            # Simple fallback splitting
            chunks = []
            start = 0
            while start < len(text):
                end = start + chunk_size
                chunks.append(text[start:end])
                start = end - chunk_overlap
            return chunks

    def process_document(self, document_path: Union[str, Path], 
                        chunk_size: int = 1000, 
                        chunk_overlap: int = 200) -> bool:
        """
        Process a document and add it to the vector store.
        """
        if not self.initialized:
            raise RuntimeError("RAG service not initialized. Call initialize() first.")
            
        try:
            # Ensure Pinecone index is ready (lazy initialization)
            if not self.index and self.pinecone_api_key and HAS_PINECONE:
                self._ensure_pinecone_index()
            
            # Extract text
            full_text = self._extract_text_from_file(document_path)
            
            # Generate document ID based on content hash for deduplication
            doc_id = hashlib.md5(full_text.encode()).hexdigest()
            
            # Check if document already exists
            if self.index:
                try:
                    # Check if the first chunk exists
                    fetch_response = self.index.fetch(ids=[f"{doc_id}_0"])
                    if fetch_response and fetch_response.vectors:
                        print(f"Document already exists in Pinecone (ID: {doc_id}). Skipping upsert.")
                        return True
                except Exception as e:
                    print(f"Error checking for existing document: {e}")

            # Split text
            chunks = self._split_text(full_text, chunk_size, chunk_overlap)
            
            # Embed and store chunks
            if self.index:
                # Use Pinecone
                print(f"Upserting to Pinecone index: {self.index_name}")
                vectors_to_upsert = []
                for i, chunk in enumerate(chunks):
                    # Add delay to avoid rate limits
                    time.sleep(1) 
                    
                    embedding = self._get_embedding(chunk)
                    if embedding:
                        vector_id = f"{doc_id}_{i}"
                        vectors_to_upsert.append({
                            "id": vector_id,
                            "values": embedding,
                            "metadata": {
                                "text": chunk,
                                "source": str(document_path),
                                "chunk_index": i,
                                "doc_id": doc_id,
                                "doc_hash": doc_id # Store hash in metadata too
                            }
                        })
                    else:
                        print(f"Failed to get embedding for chunk {i}")
                
                if not vectors_to_upsert:
                    print("No embeddings generated. Aborting upsert.")
                    return False

                # Upsert in batches
                batch_size = 50 # Reduced batch size
                for i in range(0, len(vectors_to_upsert), batch_size):
                    batch = vectors_to_upsert[i:i+batch_size]
                    try:
                        self.index.upsert(vectors=batch)
                        print(f"Upserted batch {i//batch_size + 1}")
                    except Exception as e:
                        print(f"Error upserting batch: {e}")
                
                print(f"Successfully added {len(vectors_to_upsert)} chunks to Pinecone")
            else:
                # Fallback to in-memory
                print("Using in-memory storage (Pinecone not available)")
                # Check for duplicates in memory
                if any(doc.get("doc_id") == doc_id for doc in self.documents):
                     print(f"Document already exists in memory (ID: {doc_id}). Skipping.")
                     return True

                for chunk in chunks:
                    time.sleep(0.5) # Slight delay for in-memory too
                    embedding = self._get_embedding(chunk)
                    if embedding:
                        self.documents.append({
                            "content": chunk,
                            "embedding": np.array(embedding),
                            "source": str(document_path),
                            "doc_id": doc_id
                        })
                print(f"Added {len(chunks)} chunks to in-memory storage")
            
            return True
            
        except Exception as e:
            raise RuntimeError(f"Failed to process document: {str(e)}")

    def query(self, question: str, top_k: int = 3) -> RAGResponse:
        """
        Query the RAG system with a question.
        """
        if not self.initialized:
            return RAGResponse(
                success=False,
                error="RAG service not initialized"
            )
            
        try:
            print(f"RAG Query: {question}", flush=True)
            # Get question embedding
            print("Getting question embedding...", flush=True)
            q_embedding = self._get_embedding(question)
            if not q_embedding:
                print("Failed to get question embedding", flush=True)
                return RAGResponse(success=False, error="Failed to embed question")
            print("Question embedding received", flush=True)
            
            # Search for similar documents
            if self.index:
                print("Querying Pinecone index...", flush=True)
                # Use Pinecone
                search_results = self.index.query(
                    vector=q_embedding,
                    top_k=top_k,
                    include_metadata=True
                )
                print(f"Pinecone query complete. Matches: {len(search_results.matches)}", flush=True)
                
                top_docs = [
                    {
                        "content": match.metadata.get("text", ""),
                        "source": match.metadata.get("source", ""),
                        "score": match.score
                    }
                    for match in search_results.matches
                ]
            else:
                # Use in-memory
                if not self.documents:
                    return RAGResponse(
                        success=False,
                        error="No documents uploaded. Please upload a document first."
                    )
                
                q_vec = np.array(q_embedding)
                similarities = []
                for doc in self.documents:
                    sim = np.dot(doc["embedding"], q_vec) / (
                        np.linalg.norm(doc["embedding"]) * np.linalg.norm(q_vec)
                    )
                    similarities.append((sim, doc))
                
                similarities.sort(key=lambda x: x[0], reverse=True)
                top_docs = [
                    {
                        "content": doc["content"],
                        "source": doc["source"],
                        "score": float(sim)
                    }
                    for sim, doc in similarities[:top_k]
                ]
            
            # Format context
            context = "\n\n".join(
                f"[Source: {doc['source']}]\n{doc['content']}"
                for doc in top_docs
            )
            
            # Generate answer
            prompt = (
                "You are a helpful financial analyst assistant. Answer the question based ONLY on the provided context. "
                "Format your response with the following structure:\n"
                "- Use **Bold Headers** for key sections.\n"
                "- Use bullet points for lists.\n"
                "- Keep it concise and professional.\n\n"
                "If the context doesn't contain enough information to answer the question, say so.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer:"
            )
            
            print("Generating content with Gemini...", flush=True)
            answer = self._generate_content(prompt)
            print(f"Generation complete. Answer length: {len(answer) if answer else 0}", flush=True)
            
            return RAGResponse(
                success=True,
                answer=answer,
                metadata={
                    "sources_used": list(set(doc["source"] for doc in top_docs)),
                    "num_sources": len(top_docs),
                    "relevance_scores": [doc["score"] for doc in top_docs]
                }
            )
            
        except Exception as e:
            return RAGResponse(
                success=False,
                error=f"Error processing query: {str(e)}"
            )

# Singleton instance
rag_service = RAGService()

def initialize_rag_service(api_key: Optional[str] = None, pinecone_api_key: Optional[str] = None, groq_api_key: Optional[str] = None) -> bool:
    try:
        rag_service.initialize(api_key, pinecone_api_key, groq_api_key)
        return True
    except Exception as e:
        print(f"RAG initialization error: {e}")
        return False

def query_rag(question: str, document_path: Optional[str] = None) -> RAGResponse:
    if document_path:
        try:
            rag_service.process_document(document_path)
        except Exception as e:
            return RAGResponse(
                success=False,
                error=f"Failed to process document: {str(e)}"
            )
    
    return rag_service.query(question)