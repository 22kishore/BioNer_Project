import json
import numpy as np
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split as sk_split
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

data_main    = load_json("train.json")
data_vennila = load_json("train_vennila.json")

print(f"train.json samples        : {len(data_main)}")
print(f"train_vennila.json samples: {len(data_vennila)}")

merged_data = data_main + data_vennila

print(f"Total merged samples      : {len(merged_data)}")

# ------------------ 3. CLEAN + NORMALIZE ------------------

def normalize_dataset(data):

    clean_data = []

    for sample in data:

        tokens   = sample["tokens"]
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
            "tokens":   tokens,
            "ner_tags": fixed_tags
        })

    return clean_data


print("\n🧹 Normalizing merged dataset...")
merged_data = normalize_dataset(merged_data)

print("Clean samples:", len(merged_data))

# ------------------ 4. CREATE DATASET (70 / 15 / 15 split) ------------------

# Step 1: hold out 15% as test
train_val_data, test_data = sk_split(merged_data, test_size=0.15, random_state=42)

# Step 2: split remaining into train (~70%) and val (~15%)
train_data, val_data = sk_split(train_val_data, test_size=0.176, random_state=42)

print(f"\n📊 Split summary:")
print(f"  Train : {len(train_data)} samples  (~70%)")
print(f"  Val   : {len(val_data)} samples  (~15%)")
print(f"  Test  : {len(test_data)} samples  (~15%)")

dataset = DatasetDict({
    "train":      Dataset.from_list(train_data),
    "validation": Dataset.from_list(val_data),
    "test":       Dataset.from_list(test_data)
})

# ------------------ 5. TOKENIZER ------------------

model_checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

print("\n🔤 Loading PubMedBERT tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def align_labels(examples):

    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=128,
        is_split_into_words=True
    )

    labels = []

    for i, label in enumerate(examples["ner_tags"]):

        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids         = []

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
tokenized_datasets = dataset.map(
    align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ------------------ 6. MODEL ------------------

print("\n🧠 Loading PubMedBERT model...")

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

    output_dir="./pubmedbert_model",

    evaluation_strategy="epoch",
    save_strategy="epoch",

    learning_rate=1e-5,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,

    gradient_accumulation_steps=2,   # Effective batch = 16

    num_train_epochs=15,

    fp16=True,

    weight_decay=0.01,

    warmup_ratio=0.1,

    logging_steps=50,

    load_best_model_at_end=True,
    metric_for_best_model="f1",

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
        "recall":    results["overall_recall"],
        "f1":        results["overall_f1"],
        "accuracy":  results["overall_accuracy"]
    }


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],   # validation used during training
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# ------------------ 8. TRAIN ------------------

print("\n🚀 PubMedBERT Training Started...\n")

trainer.train()

# ------------------ 9. FINAL TEST EVALUATION ------------------

print("\n🧪 Running final evaluation on held-out test set...")

# Get predictions
test_results = trainer.predict(tokenized_datasets["test"])

# Get loss separately
test_loss_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
test_loss         = test_loss_results["eval_loss"]

# Decode predictions
test_predictions = np.argmax(test_results.predictions, axis=2)
test_label_ids   = test_results.label_ids

true_predictions = [
    [label_list[p] for (p, l) in zip(pred, labels) if l != -100]
    for pred, labels in zip(test_predictions, test_label_ids)
]
true_labels = [
    [label_list[l] for (p, l) in zip(pred, labels) if l != -100]
    for pred, labels in zip(test_predictions, test_label_ids)
]

# Compute seqeval metrics
test_metrics = seqeval.compute(
    predictions=true_predictions,
    references=true_labels
)

# Overall metrics
print("\n📈 ── Test Set Results ──────────────────────────")
print(f"  Loss      : {test_loss:.4f}")
print(f"  Precision : {test_metrics['overall_precision']:.4f}")
print(f"  Recall    : {test_metrics['overall_recall']:.4f}")
print(f"  F1 Score  : {test_metrics['overall_f1']:.4f}")
print(f"  Accuracy  : {test_metrics['overall_accuracy']:.4f}")

# Per-entity breakdown
print("\n📋 ── Per-Entity Report ─────────────────────────")
for entity, scores in test_metrics.items():
    if isinstance(scores, dict):
        print(f"  {entity:<20}  P={scores['precision']:.4f}  R={scores['recall']:.4f}  F1={scores['f1']:.4f}  Support={scores['number']}")

# Save metrics to file
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open("./test_metrics.json", "w") as f:
    json.dump(test_metrics, f, indent=2, cls=NumpyEncoder)

print("\n💾 Test metrics saved to test_metrics.json")

# ------------------ 10. SAVE FINAL MODEL ------------------

print("\n💾 Saving best PubMedBERT model...")

trainer.save_model("./pubmedbert_model_kishore_vennila_merged")
tokenizer.save_pretrained("./pubmedbert_model_kishore_vennila_merged")

print("\n✅ PubMedBERT Training Completed Successfully!")