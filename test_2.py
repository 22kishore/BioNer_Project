from transformers import pipeline
import json
import numpy as np
from collections import defaultdict

# =====================================================
# 1️⃣ Custom JSON Encoder — fixes float32 error
# =====================================================
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# =====================================================
# 2️⃣ Load Model
# =====================================================
print("Loading model...")
nlp = pipeline(
    "ner",
    model="./pubmedbert_model_kishore_data",
    tokenizer="./pubmedbert_model_kishore_data",
    aggregation_strategy="simple",
    device=0
)

# =====================================================
# 3️⃣ Your 5 Abstracts
# =====================================================
documents = [
    {
        "id": "LPMO_TEST_001",
        "title": "Mucinolysome in gut microbiomes of farm animals and humans",
        "text": (
            "Mucins are glycoproteins that create a protective barrier protecting host tissues "
            "from microbial pathogens and are instrumental for host health. Here, we provide "
            "evidence that mucin glycan degradation in the gut can be mediated by mucinolysomes, "
            "defined as extracellular multi-enzyme complexes specializing in mucin glycan degradation. "
            "We computationally predicted the presence of mucinolysomes across 63 metagenome-assembled "
            "genomes (MAGs) and two isolated genomes of anaerobic Limousia bacteria, including seven "
            "MAGs from human samples of six countries. All 65 genomes were found to display core "
            "mucinolysome components, consisting of 3~6 scaffoldins (containing up to 12 cohesin modules) "
            "and up to 22 dockerin-containing mucin glycan-degrading CAZymes (carbohydrate active enzymes). "
            "The organization of mucinolysomes allows the assembly of up to 24 CAZymes in the same complex. "
            "We validated that a cultivated Limousia strain ET540 from chicken cecum can support growth on "
            "mucins as its sole carbon source, triggering the expression of most mucinolysome-related genes, "
            "including both scaffoldins and CAZymes. We also modeled the assembly of proteins into a "
            "multi-enzyme complex by predicting the cohesin-dockerin interactions among most of the "
            "mucinolysome proteins using AlphaFold3."
        )
    },
    {
        "id": "LPMO_TEST_002",
        "title": "Tripartite binding mode of cohesin-dockerin complexes from Ruminococcus flavefaciens",
        "text": (
            "Polysaccharides in plant cell walls serve as a rich carbon and energy source, yet their "
            "structural complexity presents a barrier to efficient degradation. To address this, anaerobic "
            "microorganisms like R. flavefaciens have developed sophisticated multi-enzyme complexes known "
            "as cellulosomes, which enable the efficient breakdown of these recalcitrant polysaccharides. "
            "These complexes are assembled through high-affinity interactions between cohesin (Coh) modules "
            "in scaffoldin proteins and dockerin (Doc) modules in cellulosomal enzymes. R. flavefaciens FD-1 "
            "harbors one of the most intricate cellulosomes described to date, comprising over 200 "
            "Doc-containing proteins encoded in its genome. Our findings reveal a novel tripartite binding "
            "mechanism, in which the cohesin can simultaneously bind two distinct dockerin units in three "
            "alternative conformations."
        )
    },
    {
        "id": "LPMO_TEST_003",
        "title": "Development of a thermophilic l-arabinose-inducible system in Acetivibrio thermocellus",
        "text": (
            "Inducible genetic operation systems constitute essential tools in microbial synthetic biology "
            "and metabolic engineering. Acetivibrio thermocellus, a representative strain of thermophilic "
            "non-model microbes, currently serves as a promising chassis organism in biorefinery. "
            "We developed a thermostable l-arabinose-inducible system (ThermoARAi) in A. thermocellus "
            "by utilizing the inducible promoter PabnE and repressor AraR from Geobacillus stearothermophilus T-6. "
            "The system exhibited dynamic range improvement from a 5.4-fold induction to a 175-fold induction "
            "with negligible leakage."
        )
    },
    {
        "id": "LPMO_TEST_004",
        "title": "Spatial constraints drive amylosome-mediated resistant starch degradation by Ruminococcus bromii",
        "text": (
            "Degradation of complex dietary fiber by gut microbes is essential for colonic fermentation, "
            "short-chain fatty acid production, and microbiome function. Ruminococcus bromii is the primary "
            "resistant starch (RS) degrader in humans, which relies on the amylosome, a specialized "
            "cell-bound enzymatic complex. Proteomics demonstrates remodeling of its contents across "
            "different growth conditions, with Amy4 and Amy16 comprising 60% of the amylosome in response "
            "to RS. Structural and biochemical analyses reveal complementarity and synergistic RS degradation "
            "by these enzymes."
        )
    },
    {
        "id": "LPMO_TEST_005",
        "title": "Deconstruction by C. thermocellum - dynamic redistribution of cellulosomes",
        "text": (
            "Clostridium thermocellum is one of the most efficient microorganisms for the deconstruction "
            "of cellulosic biomass. C. thermocellum uses large multienzyme complexes known as cellulosomes "
            "to break down complex polysaccharides, notably cellulose, found in plant cell walls. "
            "The attachment of bacterial cells to the nearby substrate via the cellulosome has been "
            "hypothesized to be the reason for this high efficiency. We show that C. thermocellum initially "
            "retains its cellulosomes primarily on the cell surface but then relocates large cellulosome "
            "clusters to the interface with biomass."
        )
    }
]

# =====================================================
# 4️⃣ Predict NER Tags for Each Document
# =====================================================
all_predictions = []

for doc in documents:
    print(f"\n{'='*65}")
    print(f"📄 {doc['id']} | {doc['title']}")
    print(f"{'='*65}")

    predictions = nlp(doc["text"])

    # Tokenize by whitespace
    tokens = doc["text"].split()

    # Map character offsets to word positions
    word_char_starts = []
    char_pos = 0
    for token in tokens:
        word_char_starts.append(char_pos)
        char_pos += len(token) + 1

    # Assign BIO tags to each token
    ner_tags = ["O"] * len(tokens)

    for pred in predictions:
        p_start = pred["start"]
        p_end   = pred["end"]
        label   = pred["entity_group"]

        for i, (w_start, token) in enumerate(zip(word_char_starts, tokens)):
            w_end = w_start + len(token)
            if w_start < p_end and w_end > p_start:
                if ner_tags[i] == "O":
                    ner_tags[i] = f"B-{label}"
                else:
                    ner_tags[i] = f"I-{label}"

    # Print token-level predictions
    print(f"\n{'TOKEN':<30} {'NER TAG':<25} {'NOTE'}")
    print("-" * 65)
    for token, tag in zip(tokens, ner_tags):
        note = "✅ ENTITY" if tag != "O" else ""
        print(f"{token:<30} {tag:<25} {note}")

    # Print entity summary
    print(f"\n📌 ENTITIES FOUND:")
    print(f"{'ENTITY':<35} {'TYPE':<20} {'SCORE'}")
    print("-" * 65)
    for pred in predictions:
        print(f"{pred['word']:<35} {pred['entity_group']:<20} {float(pred['score']):.4f}")

    # ✅ FIX: cast all numpy types to native Python types before saving
    all_predictions.append({
        "id":       doc["id"],
        "title":    doc["title"],
        "tokens":   tokens,
        "ner_tags": ner_tags,
        "entities": [
            {
                "word":  p["word"],
                "label": p["entity_group"],
                "score": float(p["score"]),   # ✅ float32 → float
                "start": int(p["start"]),     # ✅ int32  → int
                "end":   int(p["end"])        # ✅ int32  → int
            }
            for p in predictions
        ]
    })

# =====================================================
# 5️⃣ Save to JSON — using NumpyEncoder as safety net
# =====================================================
output_path = "./predicted_ner_tags.json"
with open(output_path, "w") as f:
    json.dump(all_predictions, f, indent=2, cls=NumpyEncoder)  # ✅ safe encoder

print(f"\n\n✅ All predictions saved to: {output_path}")

# =====================================================
# 6️⃣ Overall Summary
# =====================================================
print("\n" + "="*65)
print("📊 OVERALL SUMMARY")
print("="*65)

entity_type_counts = defaultdict(int)
total_entities = 0

for doc in all_predictions:
    for ent in doc["entities"]:
        entity_type_counts[ent["label"]] += 1
        total_entities += 1

print(f"\n{'ENTITY TYPE':<25} {'COUNT':>8}")
print("-" * 35)
for label, count in sorted(entity_type_counts.items(), key=lambda x: -x[1]):
    print(f"{label:<25} {count:>8}")
print("-" * 35)
print(f"{'TOTAL':<25} {total_entities:>8}")
print(f"\n📄 Documents processed : {len(all_predictions)}")