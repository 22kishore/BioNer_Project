import json
from collections import Counter
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
import torch
import torch.nn as nn

# ---------------- GPU ----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

# ---------------- LABELS ----------------

label_list = [
    "O",
    "B-ENZYME", "I-ENZYME",
    "B-SUBSTRATE", "I-SUBSTRATE",
    "B-RELATION", "I-RELATION"
]

label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for i, l in enumerate(label_list)}
num_labels = len(label_list)

# ---------------- LOAD DATA ----------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

data1 = load_json("train.json")
data2 = load_json("train_vennila.json")

merged_data = data1 + data2

# ---------------- CLEAN DATA ----------------

def normalize_dataset(data):

    clean = []

    for s in data:
        tokens = [str(t) for t in s["tokens"]]
        ner_tags = s["ner_tags"]

        fixed_tags = []
        for tag in ner_tags:
            if isinstance(tag, int):
                tag = id2label[tag]
            if "MICROBE" in tag or tag not in label2id:
                fixed_tags.append("O")
            else:
                fixed_tags.append(tag)

        if len(tokens) == len(fixed_tags):
            clean.append({"tokens": tokens, "ner_tags": fixed_tags})

    return clean

merged_data = normalize_dataset(merged_data)
print(f"Clean samples: {len(merged_data)}")

# ---------------- TRAIN / TEST SPLIT ----------------

dataset = Dataset.from_list(merged_data).train_test_split(
    test_size=0.2,
    seed=42
)

# ---------------- CLASS WEIGHTS ----------------

counter = Counter()
for s in merged_data:
    counter.update(s["ner_tags"])

total = sum(counter.values())
weights = [total / counter[label] for label in label_list]
class_weights = torch.tensor(weights).to(device)

print("📊 Class weights:", class_weights)

# ---------------- TOKENIZER ----------------

model_checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align(examples):

    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    labels = []
    for i, ner_tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != prev:
                label_ids.append(label2id[ner_tags[word_id]])
            else:
                label_ids.append(-100)
            prev = word_id

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized

tokenized_ds = dataset.map(
    tokenize_and_align,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ---------------- MODEL ----------------

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
).to(device)

# ---------------- CUSTOM TRAINER (WEIGHTED LOSS) ----------------

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = nn.CrossEntropyLoss(
            weight=class_weights,
            ignore_index=-100
        )

        loss = loss_fct(
            logits.view(-1, num_labels),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss

# ---------------- TRAINING CONFIG ----------------

training_args = TrainingArguments(

    output_dir="./pubmedbert_weighted_model",  # 🔥 NEW MODEL NAME

    evaluation_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,

    num_train_epochs=5,

    fp16=True,

    warmup_ratio=0.1,
    weight_decay=0.01,

    logging_steps=50,

    save_total_limit=2,

    load_best_model_at_end=True,

    report_to="none"
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ---------------- TRAIN ----------------

print("\n🚀 PubMedBERT (Class-Weighted) Training Started\n")
trainer.train()

# ---------------- SAVE ----------------

trainer.save_model("./pubmedbert_weighted_model")
tokenizer.save_pretrained("./pubmedbert_weighted_model")

print("\n✅ Training Finished Successfully!")
