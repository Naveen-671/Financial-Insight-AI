import os
import re
import uuid
import json
import requests
from pathlib import Path
from typing import Dict, Any, List, Optional
from config import Config


class NERService:
    """NER Service using HuggingFace Inference API instead of local models."""
    
    def __init__(self):
        self.model_loaded = False
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY", "")
        self.hf_model = "dslim/bert-base-NER"  # Popular NER model on HuggingFace
        self.api_url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        Config.create_directories()
        
        if self.hf_api_key:
            self.model_loaded = True
            print(f"NER Service initialized with HuggingFace Inference API (model: {self.hf_model})")
        else:
            print("Warning: HUGGINGFACE_API_KEY not set. NER will use basic regex fallback.")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Initialize NER service (API-based, no local model loading needed).
        
        Args:
            model_path: Ignored for API-based approach
            
        Returns:
            True if API key is available, False otherwise
        """
        if self.hf_api_key:
            self.model_loaded = True
            return True
        
        # Even without API key, we can use regex fallback
        self.model_loaded = True
        return True
    
    def _call_hf_api(self, text: str) -> List[Dict[str, Any]]:
        """Call HuggingFace Inference API for NER."""
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json={"inputs": text},
                timeout=30
            )
            
            if response.status_code == 503:
                # Model is loading, wait and retry
                import time
                time.sleep(5)
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json={"inputs": text},
                    timeout=60
                )
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"HuggingFace API error: {e}")
            return []
    
    def _regex_fallback_ner(self, text: str) -> List[Dict[str, Any]]:
        """Basic regex-based NER fallback when API is unavailable."""
        entities = []
        
        # Money/Currency patterns
        money_pattern = r'\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|trillion|M|B|K))?|\d+(?:\.\d+)?\s*(?:USD|EUR|GBP|INR|JPY)'
        for match in re.finditer(money_pattern, text, re.IGNORECASE):
            entities.append({
                "text": match.group(),
                "label": "MONEY",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.8
            })
        
        # Percentage patterns
        pct_pattern = r'\d+(?:\.\d+)?%'
        for match in re.finditer(pct_pattern, text):
            entities.append({
                "text": match.group(),
                "label": "PERCENT",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.9
            })
        
        # Date patterns
        date_pattern = r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b'
        for match in re.finditer(date_pattern, text, re.IGNORECASE):
            entities.append({
                "text": match.group(),
                "label": "DATE",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.85
            })
        
        # Organization-like patterns (capitalized multi-word sequences)
        org_pattern = r'\b(?:[A-Z][a-z]+\s+){1,3}(?:Inc|Corp|Ltd|LLC|Group|Bank|Fund|Capital|Holdings|Securities|Partners|Associates|Company|Co)\b\.?'
        for match in re.finditer(org_pattern, text):
            entities.append({
                "text": match.group().strip(),
                "label": "ORG",
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.7
            })
        
        return entities
    
    def _map_hf_label(self, label: str) -> str:
        """Map HuggingFace NER labels to standard labels."""
        label_map = {
            "B-PER": "PERSON", "I-PER": "PERSON",
            "B-ORG": "ORG", "I-ORG": "ORG",
            "B-LOC": "LOC", "I-LOC": "LOC",
            "B-MISC": "MISC", "I-MISC": "MISC",
        }
        return label_map.get(label, label)
    
    def _merge_entities(self, raw_entities: List[Dict]) -> List[Dict[str, Any]]:
        """Merge consecutive sub-token entities from HuggingFace API."""
        if not raw_entities:
            return []
        
        merged = []
        current = None
        
        for entity in raw_entities:
            label = self._map_hf_label(entity.get("entity_group", entity.get("entity", "")))
            word = entity.get("word", "")
            start = entity.get("start", 0)
            end = entity.get("end", 0)
            score = entity.get("score", 0.0)
            
            # Check if this continues the previous entity
            if current and label == current["label"] and start <= current["end"] + 2:
                # Merge with current entity
                current["text"] = current["text"] + word.replace("##", "")
                current["end"] = end
                current["confidence"] = min(current["confidence"], score)
            else:
                # Save previous entity
                if current:
                    merged.append(current)
                
                current = {
                    "text": word.replace("##", ""),
                    "label": label,
                    "start": start,
                    "end": end,
                    "confidence": round(score, 4)
                }
        
        if current:
            merged.append(current)
        
        return merged
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract named entities from text using HuggingFace Inference API.
        
        Args:
            text: Input text for NER processing
            
        Returns:
            Dictionary containing entities and metadata
        """
        try:
            if not self.model_loaded:
                self.load_model()
            
            entities = []
            method_used = "regex_fallback"
            
            # Try HuggingFace API first
            if self.hf_api_key:
                # HF API has input length limits, process in chunks
                max_chunk_size = 1000
                chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
                
                all_raw_entities = []
                offset = 0
                
                for chunk in chunks:
                    raw_entities = self._call_hf_api(chunk)
                    
                    if isinstance(raw_entities, list) and raw_entities:
                        # Adjust offsets for chunked processing
                        for ent in raw_entities:
                            if isinstance(ent, dict):
                                ent["start"] = ent.get("start", 0) + offset
                                ent["end"] = ent.get("end", 0) + offset
                                all_raw_entities.append(ent)
                    
                    offset += len(chunk)
                
                if all_raw_entities:
                    entities = self._merge_entities(all_raw_entities)
                    method_used = f"huggingface_api ({self.hf_model})"
                else:
                    # Fallback to regex
                    entities = self._regex_fallback_ner(text)
                    method_used = "regex_fallback (API returned empty)"
            else:
                # No API key, use regex fallback
                entities = self._regex_fallback_ner(text)
            
            # Generate visualization HTML
            html_content = self._generate_visualization_html(text, entities)
            
            # Save HTML file
            output_filename = f"ner_entities_{uuid.uuid4().hex}.html"
            output_path = Config.OUTPUTS_DIR / output_filename
            
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            return {
                "success": True,
                "entities": entities,
                "entity_count": len(entities),
                "visualization_file": output_filename,
                "visualization_url": f"/outputs/{output_filename}",
                "metadata": {
                    "text_length": len(text),
                    "method": method_used,
                    "entity_types": list(set(e["label"] for e in entities))
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "entities": [],
                "entity_count": 0
            }
    
    def _generate_visualization_html(self, text: str, entities: List[Dict[str, Any]]) -> str:
        """Generate HTML visualization similar to spaCy's displacy."""
        # Sort entities by start position
        sorted_entities = sorted(entities, key=lambda x: x.get("start", 0))
        
        # Color map for entity types
        colors = {
            "PERSON": "#aa9cfc", "PER": "#aa9cfc",
            "ORG": "#7aecec",
            "LOC": "#feca74", "GPE": "#feca74",
            "MISC": "#c887fb",
            "MONEY": "#e4e7d2",
            "PERCENT": "#e4e7d2",
            "DATE": "#bfe1d9",
            "TIME": "#bfe1d9",
            "CARDINAL": "#e4e7d2",
        }
        
        html_parts = []
        html_parts.append("""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>NER Visualization</title>
<style>
body { font-family: 'Inter', Arial, sans-serif; padding: 40px; line-height: 1.8; color: #1e293b; max-width: 900px; margin: 0 auto; }
.entity { padding: 2px 6px; margin: 0 2px; border-radius: 4px; display: inline; font-weight: 500; }
.entity-label { font-size: 0.7em; font-weight: 700; margin-left: 6px; vertical-align: middle; text-transform: uppercase; }
h1 { color: #0f172a; border-bottom: 3px solid #3b82f6; padding-bottom: 10px; }
</style></head><body>
<h1>Named Entity Recognition Results</h1>
<div style="font-size: 16px; line-height: 2;">
""")
        
        current_pos = 0
        for ent in sorted_entities:
            start = ent.get("start", 0)
            end = ent.get("end", 0)
            label = ent.get("label", "UNKNOWN")
            color = colors.get(label, "#ddd")
            
            # Add text before entity
            if start > current_pos:
                html_parts.append(text[current_pos:start])
            
            # Add entity with highlighting
            entity_text = text[start:end] if end <= len(text) else ent.get("text", "")
            html_parts.append(
                f'<span class="entity" style="background: {color}">'
                f'{entity_text}'
                f'<span class="entity-label">{label}</span>'
                f'</span>'
            )
            
            current_pos = end
        
        # Add remaining text
        if current_pos < len(text):
            html_parts.append(text[current_pos:])
        
        html_parts.append("</div></body></html>")
        
        return "".join(html_parts)
    
    def get_entity_statistics(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about extracted entities.
        
        Args:
            entities: List of entity dictionaries
            
        Returns:
            Entity statistics
        """
        if not entities:
            return {"total_entities": 0, "entity_types": {}}
        
        entity_types = {}
        for entity in entities:
            label = entity.get("label", "UNKNOWN")
            entity_types[label] = entity_types.get(label, 0) + 1
        
        return {
            "total_entities": len(entities),
            "entity_types": entity_types,
            "unique_types": len(entity_types)
        }
    
    def is_model_available(self) -> bool:
        """
        Check if NER service is available.
        
        Returns:
            bool: True if service is available (always true, uses fallback)
        """
        if not self.model_loaded:
            return self.load_model()
        return True
