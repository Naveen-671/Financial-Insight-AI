import os
import uuid
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from config import Config


class FinBERTService:
    """FinBERT Sentiment Analysis using HuggingFace Inference API."""
    
    def __init__(self):
        self.model_loaded = False
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        self.hf_model = "ProsusAI/finbert"
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{self.hf_model}"
        self.device = "api"
        Config.create_directories()

        # Download NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            print(f"Warning: NLTK download failed: {e}")
        
        if self.hf_api_key:
            self.model_loaded = True
            print(f"FinBERT Service initialized with HuggingFace Inference API")
        else:
            print("Warning: HUGGINGFACE_API_KEY not set. Sentiment analysis will not work.")

    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Initialize FinBERT service (API-based, no local model loading needed).

        Args:
            model_name: Ignored for API-based approach

        Returns:
            True if API key is available, False otherwise
        """
        if self.hf_api_key:
            self.model_loaded = True
            return True
        
        print("Error: HUGGINGFACE_API_KEY not set")
        return False

    def _call_hf_api(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Call HuggingFace Inference API for sentiment analysis."""
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        
        results = []
        for text in texts:
            try:
                # Truncate text to 512 chars to respect model limits
                truncated_text = text[:512]
                
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json={"inputs": truncated_text},
                    timeout=30
                )
                
                if response.status_code == 503:
                    # Model is loading, wait and retry
                    import time
                    time.sleep(10)
                    response = requests.post(
                        self.api_url,
                        headers=headers,
                        json={"inputs": truncated_text},
                        timeout=60
                    )
                
                response.raise_for_status()
                result = response.json()
                
                # HF API returns [[{label, score}, ...]] for classification
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], list):
                        # Get the top prediction
                        top_pred = max(result[0], key=lambda x: x.get("score", 0))
                        results.append({
                            "label": top_pred.get("label", "neutral").lower(),
                            "score": top_pred.get("score", 0.0)
                        })
                    elif isinstance(result[0], dict):
                        top_pred = max(result, key=lambda x: x.get("score", 0))
                        results.append({
                            "label": top_pred.get("label", "neutral").lower(),
                            "score": top_pred.get("score", 0.0)
                        })
                    else:
                        results.append({"label": "neutral", "score": 0.0, "error": "Unexpected response format"})
                else:
                    results.append({"label": "neutral", "score": 0.0, "error": "Empty response"})
                    
            except Exception as e:
                print(f"HuggingFace API error for text '{text[:50]}...': {e}")
                results.append({"label": "neutral", "score": 0.0, "error": str(e)})
        
        return results

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using FinBERT via HuggingFace API.

        Args:
            text: Input text for sentiment analysis

        Returns:
            Dictionary containing sentiment results
        """
        try:
            if not self.model_loaded and not self.load_model():
                return {
                    "success": False,
                    "error": "Failed to initialize FinBERT service (missing API key)",
                    "text": text
                }

            # Process text
            is_batch = isinstance(text, list)
            texts = text if is_batch else [text]

            results = self._call_hf_api(texts)

            if is_batch:
                return {
                    "success": True,
                    "results": [{
                        "label": r.get("label", "neutral"),
                        "score": r.get("score", 0.0),
                        "text": texts[i] if i < len(texts) else ""
                    } for i, r in enumerate(results)],
                    "device": self.device
                }
            else:
                if results and "error" not in results[0]:
                    return {
                        "success": True,
                        "label": results[0].get("label", "neutral"),
                        "score": results[0].get("score", 0.0),
                        "text": text,
                        "device": self.device
                    }
                else:
                    return {
                        "success": False,
                        "error": results[0].get("error", "Unknown error") if results else "No results",
                        "text": text
                    }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "text": text
            }

    def analyze_document_sentiment(self, html_content: str) -> Dict[str, Any]:
        """
        Analyze sentiment for each sentence in HTML document.

        Args:
            html_content: HTML content of the document

        Returns:
            Dictionary containing sentiment analysis and highlighted HTML
        """
        try:
            if not self.model_loaded:
                if not self.load_model():
                    raise RuntimeError("Failed to initialize FinBERT service")

            # Parse HTML
            soup = BeautifulSoup(html_content, "html.parser")
            paragraphs = soup.find_all("p")

            sentence_results = []
            total_sentences = 0

            # Process each paragraph
            for p in paragraphs:
                paragraph_text = p.get_text(strip=True)
                if not paragraph_text:
                    continue

                sentences = sent_tokenize(paragraph_text)
                highlighted_sentences = []

                for sent in sentences:
                    # Analyze sentiment
                    sentiment_result = self.analyze_sentiment(sent)

                    if sentiment_result["success"]:
                        label = sentiment_result["label"].lower()
                        score = sentiment_result["score"]
                    else:
                        label = "neutral"
                        score = 0.0

                    # Get color for highlighting
                    color = Config.SENTIMENT_COLORS.get(label, "#ffffff")

                    # Create highlighted sentence
                    highlighted_sent = f'<span style="background-color:{color}; padding:2px; border-radius:2px;" title="{label}: {score:.2f}">{sent}</span>'
                    highlighted_sentences.append(highlighted_sent)

                    # Store result
                    sentence_results.append({
                        "text": sent,
                        "label": label,
                        "score": score,
                        "paragraph_index": total_sentences
                    })

                # Replace paragraph content with highlighted sentences
                p.clear()
                p.append(BeautifulSoup(
                    " ".join(highlighted_sentences), "html.parser"))
                total_sentences += len(sentences)

            # Generate final HTML
            highlighted_html = soup.prettify()

            # Save HTML file
            output_filename = f"sentiment_analysis_{uuid.uuid4().hex}.html"
            output_path = Config.OUTPUTS_DIR / output_filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(highlighted_html)

            # Calculate statistics
            sentiment_stats = self._calculate_sentiment_stats(sentence_results)

            return {
                "success": True,
                "sentence_results": sentence_results,
                "statistics": sentiment_stats,
                "total_sentences": total_sentences,
                "highlighted_html_file": output_filename,
                "highlighted_html_url": f"/outputs/{output_filename}",
                "metadata": {
                    "model_used": self.hf_model,
                    "inference_method": "huggingface_api",
                    "sentiment_colors": Config.SENTIMENT_COLORS
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sentence_results": [],
                "statistics": {},
                "total_sentences": 0
            }

    def _calculate_sentiment_stats(self, sentence_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate sentiment statistics from sentence results."""
        if not sentence_results:
            return {}

        # Count sentiment labels
        sentiment_counts = {}
        sentiment_scores = {}

        for result in sentence_results:
            label = result.get("label", "neutral")
            score = result.get("score", 0.0)

            sentiment_counts[label] = sentiment_counts.get(label, 0) + 1

            if label not in sentiment_scores:
                sentiment_scores[label] = []
            sentiment_scores[label].append(score)

        # Calculate averages
        sentiment_averages = {}
        for label, scores in sentiment_scores.items():
            sentiment_averages[label] = sum(scores) / len(scores)

        # Determine overall sentiment
        overall_sentiment = max(
            sentiment_counts.items(), key=lambda x: x[1])[0]

        return {
            "sentiment_distribution": sentiment_counts,
            "average_scores": sentiment_averages,
            "overall_sentiment": overall_sentiment,
            "total_sentences_analyzed": len(sentence_results)
        }

    def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment for plain text (sentence by sentence).

        Args:
            text: Plain text input

        Returns:
            Dictionary containing sentiment analysis
        """
        try:
            if not self.model_loaded:
                if not self.load_model():
                    raise RuntimeError("Failed to initialize FinBERT service")

            # Tokenize sentences
            sentences = sent_tokenize(text)
            sentence_results = []

            for i, sent in enumerate(sentences):
                sentiment_result = self.analyze_sentiment(sent)

                if sentiment_result["success"]:
                    sentence_results.append({
                        "text": sent,
                        "label": sentiment_result["label"].lower(),
                        "score": sentiment_result["score"],
                        "sentence_index": i
                    })
                else:
                    sentence_results.append({
                        "text": sent,
                        "label": "neutral",
                        "score": 0.0,
                        "sentence_index": i,
                        "error": sentiment_result.get("error")
                    })

            # Calculate statistics
            sentiment_stats = self._calculate_sentiment_stats(sentence_results)

            return {
                "success": True,
                "sentence_results": sentence_results,
                "statistics": sentiment_stats,
                "total_sentences": len(sentences),
                "metadata": {
                    "model_used": self.hf_model,
                    "inference_method": "huggingface_api",
                    "text_length": len(text)
                }
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sentence_results": [],
                "statistics": {},
                "total_sentences": 0
            }

    def is_model_available(self) -> bool:
        """Check if FinBERT service is available."""
        return bool(self.hf_api_key) and self.model_loaded

    def get_sentiment_colors(self) -> Dict[str, str]:
        """Get sentiment color mapping."""
        return Config.SENTIMENT_COLORS.copy()
