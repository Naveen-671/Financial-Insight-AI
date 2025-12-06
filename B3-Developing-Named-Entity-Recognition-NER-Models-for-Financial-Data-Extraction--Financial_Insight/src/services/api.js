// API Service for Financial Insight Backend Integration
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001';

/**
 * Process a document through the full pipeline
 * @param {File} file - The file to process
 * @param {Object} options - Processing options (include_ner, include_finbert, include_langextract)
 * @returns {Promise<Object>} - Processing results
 */
export async function processDocument(file, options = {}) {
  const formData = new FormData();
  formData.append('file', file);

  // Add options as individual form fields
  formData.append('include_ner', options.ner !== false); // Default to true if not specified
  formData.append('include_finbert', options.sentiment !== false);
  formData.append('include_langextract', options.clauses !== false);

  // Add optional model configurations if present
  if (options.nerModelPath) formData.append('ner_model_path', options.nerModelPath);
  if (options.finbertModelName) formData.append('finbert_model_name', options.finbertModelName);
  if (options.langextractModels) formData.append('langextract_models', options.langextractModels);
  if (options.langextractApiKey) formData.append('langextract_api_key', options.langextractApiKey);
  if (options.clausePrompt) formData.append('clauses_prompt', options.clausePrompt);

  const response = await fetch(`${API_BASE_URL}/api/v1/pipeline/process-document`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Analyze sentiment of text
 * @param {string} text - Text to analyze
 * @returns {Promise<Object>} - Sentiment analysis results
 */
export async function analyzeSentiment(text) {
  const response = await fetch(`${API_BASE_URL}/api/v1/finbert/analyze-sentiment`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ text }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Extract clauses from text
 * @param {string} text - Text to extract clauses from
 * @param {Array} models - Optional list of models to use
 * @returns {Promise<Object>} - Extraction results
 */
export async function extractClauses(text, models = null) {
  const body = { text };
  if (models) body.models = models;

  const response = await fetch(`${API_BASE_URL}/api/v1/langextract/extract-clauses`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

// ===== RAG FUNCTIONS =====

/**
 * Upload a document to RAG system
 * @param {File} file - The document file to upload
 * @returns {Promise<Object>} - Upload result
 */
export async function uploadRAGDocument(file) {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/v1/rag/upload-document`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Query the RAG system
 * @param {string} question - The question to ask
 * @param {number} top_k - Number of results to retrieve
 * @returns {Promise<Object>} - Query response
 */
export async function queryRAG(question, top_k = 3) {
  const response = await fetch(`${API_BASE_URL}/api/v1/rag/query`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ question, top_k }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Get RAG system status
 * @returns {Promise<Object>} - RAG status
 */
export async function getRAGStatus() {
  const response = await fetch(`${API_BASE_URL}/api/v1/rag/status`);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

// ===== EXPORT FUNCTIONS =====

/**
 * Export analysis results
 * @param {Object} data - Analysis results to export
 * @param {string} format - Export format (json, csv, txt)
 * @returns {Promise<Blob>} - File blob for download
 */
export async function exportResults(data, format = 'json') {
  const response = await fetch(`${API_BASE_URL}/api/v1/export/results`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ data, format }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  return await response.blob();
}

/**
 * Generate comprehensive report
 * @param {Object} data - Analysis results
 * @returns {Promise<Blob>} - HTML report blob
 */
export async function generateReport(data) {
  const response = await fetch(`${API_BASE_URL}/api/v1/export/generate-report`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  return await response.blob();
}

/**
 * Download a blob as a file
 * @param {Blob} blob - The blob to download
 * @param {string} filename - Name for the downloaded file
 */
export function downloadBlob(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  window.URL.revokeObjectURL(url);
  document.body.removeChild(a);
}

/**
 * Check backend health
 * @returns {Promise<Object>} - Health status
 */
export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/`);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Get pipeline status
 * @returns {Promise<Object>} - Pipeline component status
 */
export async function getPipelineStatus() {
  const response = await fetch(`${API_BASE_URL}/api/v1/pipeline/pipeline-status`);

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}
