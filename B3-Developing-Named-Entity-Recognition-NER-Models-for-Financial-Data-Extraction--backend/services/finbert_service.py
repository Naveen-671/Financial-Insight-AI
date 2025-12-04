import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import nltk
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup
from config import Config

try:
    import torch
    from transformers import pipeline
    from huggingface_hub import InferenceClient
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Install with: pip install transformers torch")


class FinBERTService:
    def __init__(self):
        self.sentiment_pipeline = None
        self.model_loaded = False
        if TRANSFORMERS_AVAILABLE:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        Config.create_directories()

        # Download NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception as e:
            print(f"Warning: NLTK download failed: {e}")

    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Load FinBERT sentiment analysis model with improved error handling

        Args:
            model_name: Model name to load, defaults to config

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("transformers library not available")

            model_to_use = model_name or Config.FINBERT_MODEL
            print(f"Loading FinBERT model: {model_to_use} on {self.device}")

            # Load pipeline with device map for better memory management
            self.sentiment_pipeline = pipeline(
                "text-classification",
                model=model_to_use,
                device=0 if self.device == "cuda" else -1,  # Use GPU if available
                truncation=True,
                max_length=512
            )
            self.model_loaded = True
            print("FinBERT model loaded successfully")
            return True

        except Exception as e:
            print(f"Error loading FinBERT model: {str(e)}")
            self.model_loaded = False
            return False

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text using FinBERT with batch processing support

        Args:
            text: Input text or list of texts for sentiment analysis

        Returns:
            Dictionary containing sentiment results
        """
        try:
            if not self.model_loaded and not self.load_model():
                return {
                    "success": False,
                    "error": "Failed to load FinBERT model",
                    "text": text
                }

            # Process text in batches if it's a list
            is_batch = isinstance(text, list)
            texts = text if is_batch else [text]

            # Process in chunks to avoid OOM errors
            batch_size = 8 if self.device == "cuda" else 4
            results = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                try:
                    batch_results = self.sentiment_pipeline(batch)
                    results.extend(batch_results)
                except Exception as e:
                    print(f"Error processing batch {i//batch_size}: {str(e)}")
                    # Add error placeholders for failed batch
                    results.extend(
                        [{"error": str(e), "label": "error", "score": 0.0}] * len(batch))

            if is_batch:
                return {
                    "success": True,
                    "results": [{
                        "label": r.get("label", "neutral").lower(),
                        "score": r.get("score", 0.0),
                        "text": texts[i] if i < len(texts) else ""
                    } for i, r in enumerate(results)],
                    "device": self.device
                }
            else:
                if results and "error" not in results[0]:
                    return {
                        "success": True,
                        "label": results[0].get("label", "neutral").lower(),
                        "score": results[0].get("score", 0.0),
                        "text": text,
                        "device": self.device
                    }
                else:
                    return {
                        "success": False,
                        "error": results[0].get("error", "Unknown error"),
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
        Analyze sentiment for each sentence in HTML document

        Args:
            html_content: HTML content of the document

        Returns:
            Dictionary containing sentiment analysis and highlighted HTML
        """
        try:
            if not self.model_loaded:
                if not self.load_model():
                    raise RuntimeError("Failed to load FinBERT model")

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
                    "model_used": Config.FINBERT_MODEL,
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
        """Calculate sentiment statistics from sentence results"""
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
        Analyze sentiment for plain text (sentence by sentence)

        Args:
            text: Plain text input

        Returns:
            Dictionary containing sentiment analysis
        """
        try:
            if not self.model_loaded:
                if not self.load_model():
                    raise RuntimeError("Failed to load FinBERT model")

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
                    "model_used": Config.FINBERT_MODEL,
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
        """Check if FinBERT model is available and loaded"""
        return TRANSFORMERS_AVAILABLE and self.model_loaded

    def get_sentiment_colors(self) -> Dict[str, str]:
        """Get sentiment color mapping"""
        return Config.SENTIMENT_COLORS.copy()
