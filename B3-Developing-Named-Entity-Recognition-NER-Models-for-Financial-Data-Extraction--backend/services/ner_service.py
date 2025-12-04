import os
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
import spacy
from spacy import displacy
from config import Config

class NERService:
    def __init__(self):
        self.nlp_model = None
        self.model_loaded = False
        Config.create_directories()
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load spaCy NER model
        
        Args:
            model_path: Path to trained model, defaults to config path
            
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # First try to load the specified model or the default custom model
            path_to_use = model_path or str(Config.NER_MODEL_PATH)
            
            # Check if path exists before trying to load
            if os.path.exists(path_to_use):
                try:
                    self.nlp_model = spacy.load(path_to_use)
                    print(f"Loaded custom NER model from: {path_to_use}")
                    self.model_loaded = True
                    return True
                except Exception as e:
                    print(f"Failed to load custom model at {path_to_use}: {e}")
                    # Fall through to default model
            else:
                print(f"Custom NER model not found at {path_to_use}, trying to load default model: {Config.NER_MODEL_NAME}")
            
            try:
                # Try to load the default model
                self.nlp_model = spacy.load(Config.NER_MODEL_NAME)
                print(f"Successfully loaded default NER model: {Config.NER_MODEL_NAME}")
                self.model_loaded = True
                return True
            except OSError:
                # If model not found, try to download it
                print(f"Default model {Config.NER_MODEL_NAME} not found, attempting to download...")
                try:
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, "-m", "spacy", "download", Config.NER_MODEL_NAME])
                    self.nlp_model = spacy.load(Config.NER_MODEL_NAME)
                    print(f"Successfully downloaded and loaded default NER model: {Config.NER_MODEL_NAME}")
                    self.model_loaded = True
                    return True
                except Exception as download_error:
                    print(f"Failed to download model: {str(download_error)}")
                    raise RuntimeError(f"Failed to load NER model. Please ensure the model is installed by running: python -m spacy download {Config.NER_MODEL_NAME}")
            
        except Exception as e:
            error_msg = f"Error loading NER model: {str(e)}"
            print(error_msg)
            self.model_loaded = False
            return False
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract named entities from text using trained NER model
        
        Args:
            text: Input text for NER processing
            
        Returns:
            Dictionary containing entities and metadata
        """
        try:
            if not self.model_loaded:
                if not self.load_model():
                    raise RuntimeError("Failed to load NER model")
            
            # Process text
            doc = self.nlp_model(text)
            
            # Extract entities
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": getattr(ent, 'confidence', None)
                })
            
            # Generate visualization HTML
            html_content = displacy.render(doc, style="ent", page=True, jupyter=False)
            
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
                    "model_path": str(Config.NER_MODEL_PATH),
                    "entity_types": list(set(ent.label_ for ent in doc.ents))
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "entities": [],
                "entity_count": 0
            }
    
    def get_entity_statistics(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about extracted entities
        
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
    
    def train_model_from_annotations(self, annotations_file: str, output_dir: str) -> Dict[str, Any]:
        """
        Train spaCy NER model from annotations file
        
        Args:
            annotations_file: Path to annotations JSON file
            output_dir: Directory to save trained model
            
        Returns:
            Training result
        """
        try:
            import json
            from spacy.tokens import DocBin
            from tqdm import tqdm
            
            # Load annotations
            with open(annotations_file, 'r') as f:
                train_data = json.load(f)
            
            # Create spaCy training data
            nlp = spacy.blank("en")
            db = DocBin()
            
            processed_count = 0
            skipped_count = 0
            
            for item in tqdm(train_data.get('annotations', [])):
                if item is None or len(item) != 2:
                    skipped_count += 1
                    continue
                
                text, annot = item
                doc = nlp.make_doc(text)
                ents = []
                
                for start, end, label in annot.get("entities", []):
                    span = doc.char_span(start, end, label=label, alignment_mode="contract")
                    if span is None:
                        skipped_count += 1
                        continue
                    ents.append(span)
                
                doc.ents = ents
                db.add(doc)
                processed_count += 1
            
            # Save training data
            training_data_path = os.path.join(output_dir, "training_data.spacy")
            os.makedirs(output_dir, exist_ok=True)
            db.to_disk(training_data_path)
            
            # Initialize config
            config_path = os.path.join(output_dir, "config.cfg")
            os.system(f"python -m spacy init config {config_path} --lang en --pipeline ner --optimize efficiency --force")
            
            # Train model
            os.system(f"python -m spacy train {config_path} --output {output_dir} --paths.train {training_data_path} --paths.dev {training_data_path}")
            
            return {
                "success": True,
                "processed_examples": processed_count,
                "skipped_examples": skipped_count,
                "model_output_dir": output_dir,
                "training_data_path": training_data_path
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def is_model_available(self) -> bool:
        """
        Check if NER model is loaded and available
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        if not self.model_loaded:
            # Try to load the model if not already loaded
            return self.load_model()
        return True and self.nlp_model is not None
