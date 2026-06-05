import json
import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import torch

# ------------------ GPU CHECK ------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

if device.type == "cuda":
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("⚠ WARNING: GPU not detected — running on CPU")

# ------------------ 1. LABEL CONFIG ------------------

label_list = [
    "O",
    "B-ENZYME", "I-ENZYME",
    "B-SUBSTRATE", "I-SUBSTRATE",
    "B-RELATION", "I-RELATION"
]

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# ------------------ 2. LOAD BOTH FILES ------------------

def load_json(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)


print("\n📂 Loading datasets...")

data1 = load_json("train.json")
data2 = load_json("train_vennila.json")

print(f"train.json samples        : {len(data1)}")
print(f"train_vennila.json samples: {len(data2)}")

merged_data = data1 + data2

print(f"Total merged samples      : {len(merged_data)}")

# ------------------ 3. CLEAN + NORMALIZE ------------------

def normalize_dataset(data):

    clean_data = []

    for sample in data:

        tokens = sample["tokens"]
        ner_tags = sample["ner_tags"]

        tokens = [str(t) for t in tokens]

        fixed_tags = []

        for tag in ner_tags:

            if isinstance(tag, int):
                tag = id2label[tag]

            if "MICROBE" in tag:
                fixed_tags.append("O")

            elif tag not in label2id:
                fixed_tags.append("O")

            else:
                fixed_tags.append(tag)

        if len(tokens) != len(fixed_tags):
            print("⚠ Skipping corrupted sample")
            continue

        clean_data.append({
            "tokens": tokens,
            "ner_tags": fixed_tags
        })

    return clean_data


print("\n🧹 Normalizing merged dataset...")
merged_data = normalize_dataset(merged_data)

print("Clean samples:", len(merged_data))

# ------------------ 4. CREATE DATASET ------------------

raw_dataset = Dataset.from_list(merged_data)

dataset = raw_dataset.train_test_split(test_size=0.2)

# ------------------ 5. TOKENIZER ------------------

model_checkpoint = "dmis-lab/biobert-v1.1"

print("\n🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def align_labels(examples):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    labels = []

    for i, label in enumerate(examples["ner_tags"]):

        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])

            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


print("\n⚙ Tokenizing...")
tokenized_datasets = dataset.map(align_labels, batched=True)

# ------------------ 6. MODEL ------------------

print("\n🧠 Loading BioBERT model...")

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id
)
model.gradient_checkpointing_enable()
model.config.use_cache = False

# ------------------ 7. TRAINING CONFIG ------------------

args = TrainingArguments(
    output_dir="biobert-enzyme-output",

    evaluation_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=2,     # REDUCED for 4GB GPU
    per_device_eval_batch_size=2,

    gradient_accumulation_steps=2,     # Keeps effective batch = 4

    num_train_epochs=30,

    fp16=True,                         # MIXED PRECISION GPU BOOST ⚡

    weight_decay=0.01,

    logging_steps=20,

    save_total_limit=2,

    dataloader_pin_memory=True,

    report_to="none"
)


data_collator = DataCollatorForTokenClassification(tokenizer)

print("\n📊 Loading evaluation metric...")
seqeval = evaluate.load("seqeval")


def compute_metrics(p):

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(
        predictions=true_predictions,
        references=true_labels
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ------------------ 8. TRAIN ------------------

print("\n🚀 Training Started on GPU...\n")

trainer.train()

# ------------------ 9. SAVE MODEL ------------------

print("\n💾 Saving model...")

trainer.save_model("./final_biobert_model")
tokenizer.save_pretrained("./final_biobert_model")

print("\n✅ Training Completed Successfully!")
