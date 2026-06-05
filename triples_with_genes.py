import json
import pandas as pd
import spacy

# =========================
# LOAD MODELS
# =========================
nlp = spacy.load("en_core_sci_scibert")   # SciSpacy model

# =========================
# LOAD YOUR JSON DATA
# =========================
def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

data1 = load_json("C:/SASTRA/BioNER_project/train.json")
data2 = load_json("C:/SASTRA/BioNER_project/train_vennila.json")

data = data1 + data2

# =========================
# SIMPLE ENTITY CLASSIFIER
# =========================
def classify_entity(text):
    text_lower = text.lower()

    if text.isupper() and len(text) > 2:
        return "GENE"
    elif "saccharomyces" in text_lower or "coli" in text_lower:
        return "ORGANISM"
    elif text_lower in ["ethanol","glucose","lactose","starch","lipids"]:
        return "CHEMICAL"
    else:
        return "OTHER"

# =========================
# EXTRACT TRIPLES FROM YOUR LABELS
# =========================
def extract_triples(tokens, tags):
    triples = []
    
    enzyme, relation, substrate = None, None, None

    for token, tag in zip(tokens, tags):
        if "ENZYME" in tag:
            enzyme = token
        elif "RELATION" in tag:
            relation = token
        elif "SUBSTRATE" in tag:
            substrate = token

    if enzyme and relation and substrate:
        triples.append((enzyme, relation, substrate))

    return triples

# =========================
# PROCESS DATA
# =========================
all_triples = []

for sample in data:
    tokens = sample["tokens"]
    tags = sample["ner_tags"]

    text = " ".join(tokens)

    # 1. Your model triples
    triples = extract_triples(tokens, tags)

    # 2. SciSpacy entities
    doc = nlp(text)
    entities = [(ent.text, classify_entity(ent.text)) for ent in doc.ents]

    # 3. Merge triples
    for (s, r, t) in triples:
        all_triples.append({
            "Source": s,
            "Relation": r,
            "Target": t,
            "Source_Type": "ENZYME",
            "Target_Type": "SUBSTRATE",
            "Sentence": text
        })

    # 4. Add Gene → Enzyme relations
    genes = [e for e in entities if e[1] == "GENE"]

    for gene in genes:
        for (s, r, t) in triples:
            all_triples.append({
                "Source": gene[0],
                "Relation": "encodes",
                "Target": s,
                "Source_Type": "GENE",
                "Target_Type": "ENZYME",
                "Sentence": text
            })

    # 5. Add Organism → Gene relations
    organisms = [e for e in entities if e[1] == "ORGANISM"]

    for org in organisms:
        for gene in genes:
            all_triples.append({
                "Source": org[0],
                "Relation": "has_gene",
                "Target": gene[0],
                "Source_Type": "ORGANISM",
                "Target_Type": "GENE",
                "Sentence": text
            })

# =========================
# SAVE FINAL CSV
# =========================
df = pd.DataFrame(all_triples)

# remove duplicates
df = df.drop_duplicates()

df.to_csv("C:/SASTRA/BioNER_project/final_knowledge_graph.csv", index=False)

print("✅ Knowledge Graph CSV created: final_knowledge_graph.csv")