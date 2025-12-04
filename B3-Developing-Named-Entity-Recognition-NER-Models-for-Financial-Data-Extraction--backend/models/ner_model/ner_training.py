import spacy
from spacy.tokens import DocBin
from tqdm import tqdm
import json
import os

def train_ner_model(annotation_path, output_dir):
    print("Loading annotations...")
    with open(annotation_path, "r", encoding="utf-8") as f:
        TRAIN_DATA = json.load(f)

    nlp = spacy.blank("en")
    db = DocBin()

    print("Converting JSON to spaCy training format...")
    for item in tqdm(TRAIN_DATA):
        text = item["text"]
        doc = nlp.make_doc(text)
        ents = []

        for ent in item["entities"]:
            start = int(ent["start"])
            end = int(ent["end"])
            label = ent["label"]
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print(f"Skipping entity: '{text[start:end]}' ({label})")
            else:
                ents.append(span)

        doc.ents = ents
        db.add(doc)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    training_file = os.path.join(output_dir, "training_data.spacy")
    db.to_disk(training_file)
    print(f"Training file saved at: {training_file}")

    print("Initializing spaCy config...")
    os.system("python -m spacy init config config.cfg --lang en --pipeline ner --optimize efficiency --force")

    print("Training model...")
    os.system(f"python -m spacy train config.cfg --output {output_dir} --paths.train {training_file} --paths.dev {training_file}")

    print("Loading trained model...")
    nlp_model = spacy.load(os.path.join(output_dir, "model-best"))

    # Save entity visualization
    print("Saving entity visualization...")
    test_text = "Market analysts at MorganEast Research, based in Singapore, forecast revenue of USD 10,000,000 by 2025."
    doc = nlp_model(test_text)

    from spacy import displacy
    html = displacy.render(doc, style="ent", page=True)

    with open("./models/entities.html", "w", encoding="utf-8") as f:
        f.write(html)

    print("NER training completed!")
    return "Training complete"

if __name__ == "__main__":
    annotation_path = "./models/annotations.json"  # your JSON file path
    output_dir = "./models/"  # output directory for trained model
    train_ner_model(annotation_path, output_dir)
