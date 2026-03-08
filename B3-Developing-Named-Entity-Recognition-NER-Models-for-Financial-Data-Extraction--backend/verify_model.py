
import spacy
import os

model_path = r"c:\Users\Dinesh R\Infosys-springboard-task\Financial Insight\B3-Developing-Named-Entity-Recognition-NER-Models-for-Financial-Data-Extraction--backend\models\ner_model\model-best"

try:
    print(f"Attempting to load model from: {model_path}")
    nlp = spacy.load(model_path)
    print("SUCCESS: Model loaded successfully.")
    print(f"Pipeline: {nlp.pipe_names}")
except Exception as e:
    print(f"ERROR: Failed to load model. {e}")
