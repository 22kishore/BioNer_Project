from transformers import pipeline

# =====================================================
# 1️⃣ Load Model
# =====================================================
print("Loading model...")
nlp = pipeline(
    "ner",
    model="./pubmedbert_model_kishore_data",
    tokenizer="./pubmedbert_model_kishore_data",
    aggregation_strategy="none",
    ignore_labels=[],
    device=0
)


# =====================================================
# 2️⃣ Chunk Function
# =====================================================
def chunk_text(text, tokenizer, max_length=512, stride=50):
    tokens = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False
    )

    input_ids = tokens["input_ids"]
    offsets = tokens["offset_mapping"]

    chunks = []
    start = 0
    usable = max_length - 2

    while start < len(input_ids):
        end = start + usable
        chunk_ids = input_ids[start:end]
        chunk_offsets = offsets[start:end]

        chunk_text = tokenizer.decode(chunk_ids)
        chunks.append((chunk_text, chunk_offsets))

        start += usable - stride

    return chunks


# =====================================================
# 3️⃣ TEXT INPUT
# =====================================================
test_id = "LPMO_TEST_001"

text = (
    "Title 1: Mucinolysome in gut microbiomes of farm animals and humans\n"
    "Abstract 1: Mucins are glycoproteins that create a protective barrier protecting host tissues from microbial pathogens and are instrumental for host health. Here, we provide evidence that mucin glycan degradation in the gut can be mediated by mucinolysomes, defined as extracellular multi-enzyme complexes specializing in mucin glycan degradation. We computationally predicted the presence of mucinolysomes across 63 metagenome-assembled genomes (MAGs) and two isolated genomes of anaerobic Limousia bacteria, including seven MAGs from human samples of six countries. All 65 genomes were found to display core mucinolysome components, consisting of 3~6 scaffoldins (containing up to 12 cohesin modules) and up to 22 dockerin-containing mucin glycan-degrading CAZymes (carbohydrate active enzymes). The organization of mucinolysomes allows the assembly of up to 24 CAZymes in the same complex. We validated that a cultivated Limousia strain ET540 from chicken cecum can support growth on mucins as its sole carbon source, triggering the expression of most mucinolysome-related genes, including both scaffoldins and CAZymes. We also modeled the assembly of proteins into a multi-enzyme complex by predicting the cohesin-dockerin interactions among most of the mucinolysome proteins using AlphaFold3. While mucinolysosome-encoding Limousia have low abundance in different animal hosts, their abundance and prevalence are higher in farm animals than in humans, highlighting a potentially important role in livestock gut ecosystems. Our findings reveal a novel mechanism of mucin glycan degradation and provide a framework to explore microbial contributions to gut health and host-microbe interactions across species.\n\n"

    "Title 2: Tripartite binding mode of cohesin-dockerin complexes from Ruminococcus flavefaciens involving naturally truncated dockerins\n"
    "Abstract 2: Polysaccharides in plant cell walls serve as a rich carbon and energy source, yet their structural complexity presents a barrier to efficient degradation. To address this, anaerobic microorganisms like R. flavefaciens have developed sophisticated multi-enzyme complexes known as cellulosomes, which enable the efficient breakdown of these recalcitrant polysaccharides. These complexes are assembled through high-affinity interactions between cohesin (Coh) modules in scaffoldin proteins and dockerin (Doc) modules in cellulosomal enzymes. R. flavefaciens FD-1 harbors one of the most intricate cellulosomes described to date, comprising over 200 Doc-containing proteins encoded in its genome. Despite substantial research on this cellulosome, the role of a group of truncated but functional dockerins, known as group-2 Docs, remains unclear. In this study, we present a detailed structural and binding analysis of a Coh-Doc complex involving the cohesin from the cell-anchoring scaffoldin ScaE and a group-2 Doc that bears only one of the two Ca+2-coordinating loops that characterise the canonical Docs. Our findings reveal a novel tripartite binding mechanism, in which the cohesin can simultaneously bind two distinct dockerin units in three alternative conformations. This discovery provides new insights into the modular versatility of the R. flavefaciens cellulosome and sheds light on the mechanisms that enhance its efficiency in polysaccharide degradation.\n\n"

    "Title 3: Development of a thermophilic l-arabinose-inducible system in Acetivibrio thermocellus (Clostridium thermocellum)\n"
    "Abstract 3: Inducible genetic operation systems constitute essential tools in microbial synthetic biology and metabolic engineering. However, inducible systems in non-model microbes, particularly thermophiles, are rarely reported. Acetivibrio thermocellus (previously termed Clostridium thermocellum), a representative strain of thermophilic non-model microbes, currently serves as a promising chassis organism in biorefinery. Although various genetic tools are available for A. thermocellus, superior thermophilic inducible systems are in high demand. In this study, we developed a thermostable l-arabinose-inducible system (ThermoARAi) in A. thermocellus by utilizing the inducible promoter PabnE and repressor AraR from Geobacillus stearothermophilus T-6. Through systematic promoter engineering and optimization of induction conditions using a thermostable β-glucuronidase as reporter, the system exhibited dynamic range improvement from a 5.4-fold induction to a 175-fold induction with negligible leakage. Furthermore, the ThermoARAi system was appropriate for use in metabolic engineering, as validated by its applications in whole-cell saccharification of cellulosic substrates and degradation of amorphous polyethylene terephthalate films. The ThermoARAi system significantly expands the genetic toolkit for precise gene expression modulation, metabolic engineering, and biotechnological applications in A. thermocellus. Importantly, this approach may also serve as a foundation for developing genetic tools in other Clostridia that play key roles in diverse ecosystems, including the gut.\n\n"

    "Title 4: Spatial constraints drive amylosome-mediated resistant starch degradation by Ruminococcus bromii in the human colon\n"
    "Abstract 4: Degradation of complex dietary fiber by gut microbes is essential for colonic fermentation, short-chain fatty acid production, and microbiome function. Ruminococcus bromii is the primary resistant starch (RS) degrader in humans, which relies on the amylosome, a specialized cell-bound enzymatic complex. To unravel its architecture, function, and the interplay among its components, we applied a holistic multilayered approach: Cryo-electron tomography reveals that the amylosome comprises a constitutive extracellular layer extending toward the RS substrate. Proteomics demonstrates remodeling of its contents across different growth conditions, with Amy4 and Amy16 comprising 60% of the amylosome in response to RS. Structural and biochemical analyses reveal complementarity and synergistic RS degradation by these enzymes. We demonstrate that amylosome composition and RS degradation are regulated at two levels: structural constraints and expression-driven shifts in enzyme proportions enforce enzyme proximity, which allows R. bromii to fine-tune its adaptation to dietary fiber and shape colonic metabolism.\n\n"

    "Title 5: Deconstruction by C. thermocellum-from microbe mediated to dynamic redistribution of cellulosomes\n"
    "Abstract 5: Clostridium thermocellum is one of the most efficient microorganisms for the deconstruction of cellulosic biomass. To achieve this high level of cellulolytic activity, C. thermocellum uses large multienzyme complexes known as cellulosomes to break down complex polysaccharides, notably cellulose, found in plant cell walls. The attachment of bacterial cells to the nearby substrate via the cellulosome has been hypothesized to be the reason for this high efficiency. The region lying between the cell and the substrate has shown great variation and dynamics that are affected by the growth stage of cells and the substrate used for growth. Here, we used both super-resolution imaging and machine-learning approaches to study the distribution of C. thermocellum cellulosomes at different stages of growth. We show that C. thermocellum initially retains its cellulosomes primarily on the cell surface but then relocates large cellulosome clusters to the interface with biomass, therefore depleting its cell surface of cellulosomes. These results indicate dynamic redistribution of cellulosomes during growth, with a functional shift toward substrate-associated degradation later during growth on biomass."
)

# (your full text stays exactly as you wrote)


# =====================================================
# 4️⃣ Process Long Text Safely
# =====================================================
tokenizer = nlp.tokenizer
chunks = chunk_text(text, tokenizer)

all_results = []

for chunk_text, _ in chunks:
    res = nlp(chunk_text)
    all_results.extend(res)

# =====================================================
# 5️⃣ Remove duplicates from overlap
# =====================================================
unique = []
seen = set()

for ent in all_results:
    key = (ent["word"], ent["start"], ent["end"], ent["entity"])
    if key not in seen:
        unique.append(ent)
        seen.add(key)

all_results = unique


# =====================================================
# 6️⃣ Pretty Print
# =====================================================
print(f"\nAnalyzing Document ID: {test_id}\n")
print(f"{'ENTITY':<25} | {'TYPE':<20} | {'CONFIDENCE'}")
print("-" * 70)

current_word = ""
current_label = ""
current_score = 0.0

for item in all_results:
    word = item['word']
    label = item['entity']
    score = item['score']

    if word.startswith("##"):
        current_word += word.replace("##", "")
        current_score = (current_score + score) / 2
    else:
        if current_word:
            print(f"{current_word:<25} | {current_label:<20} | {current_score:.4f}")

        current_word = word
        current_label = label
        current_score = score

if current_word:
    print(f"{current_word:<25} | {current_label:<20} | {current_score:.4f}")
