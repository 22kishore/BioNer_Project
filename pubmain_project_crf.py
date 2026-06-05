import json
import evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    BertModel
)
import torch
import torch.nn as nn
from TorchCRF import CRF
import numpy as np

# ------------------ GPU CHECK ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

# ------------------ LABEL CONFIG ------------------
label_list = [
    "O",
    "B-ENZYME", "I-ENZYME",
    "B-SUBSTRATE", "I-SUBSTRATE",
    "B-RELATION", "I-RELATION"
]
id2label = {i: l for i, l in enumerate(label_list)}
label2id = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)

# ------------------ LOAD DATA ------------------
def load_json(file):
    with open(file, "r", encoding="utf-8") as f:
        return json.load(f)

data1 = load_json("train.json")
data2 = load_json("train_vennila.json")
merged_data = data1 + data2

def normalize_dataset(data):
    clean_data = []
    for s in data:
        tokens = [str(t) for t in s["tokens"]]
        ner_tags = s["ner_tags"]
        fixed = []
        for tag in ner_tags:
            if isinstance(tag, int):
                tag = id2label[tag]
            if "MICROBE" in tag or tag not in label2id:
                fixed.append("O")
            else:
                fixed.append(tag)
        if len(tokens) == len(fixed):
            clean_data.append({"tokens": tokens, "ner_tags": fixed})
    return clean_data

merged_data = normalize_dataset(merged_data)
dataset = Dataset.from_list(merged_data).train_test_split(test_size=0.2, seed=42)

# ------------------ TOKENIZER ------------------
model_checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align(examples):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=256,
        is_split_into_words=True
    )
    labels = []
    for i, lab in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        label_ids = []
        for w in word_ids:
            if w is None:
                label_ids.append(-100)
            elif w != prev:
                label_ids.append(label2id[lab[w]])
            else:
                label_ids.append(-100)
            prev = w
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

tokenized_ds = dataset.map(
    tokenize_and_align,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ------------------ MODEL ------------------
class PubMedBERT_CRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)
        mask = attention_mask.bool()

        if labels is not None:
            # Training mode: Calculate Loss
            labels_clean = labels.clone()
            labels_clean[labels_clean == -100] = 0
            log_likelihood = self.crf(emissions, labels_clean, mask=mask)
            return {"loss": -log_likelihood.mean()}
        else:
            # Eval mode: Decode and Pad for Trainer consistency
            preds = self.crf.decode(emissions, mask=mask)
            max_len = input_ids.shape[1]
            # Pad sequences to max_len so they can be stacked into a tensor
            padded_preds = [p + [-1] * (max_len - len(p)) for p in preds]
            return {"logits": torch.tensor(padded_preds)}

model = PubMedBERT_CRF(model_checkpoint, num_labels).to(device)

# ------------------ METRICS ------------------
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    # p.predictions are the padded sequences from CRF decode
    predictions = p.predictions
    labels = p.label_ids

    true_predictions = []
    true_labels = []

    for pred_seq, label_seq in zip(predictions, labels):
        temp_pred = []
        temp_label = []
        for p_i, l_i in zip(pred_seq, label_seq):
            if l_i != -100:
                # Use 0 (O) as fallback if CRF index is -1
                pred_idx = int(p_i) if p_i != -1 else 0
                temp_pred.append(label_list[pred_idx])
                temp_label.append(label_list[l_i])
        
        if temp_label: # Avoid empty lists
            true_predictions.append(temp_pred)
            true_labels.append(temp_label)

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"]
    }

# ------------------ TRAINING CONFIG ------------------
args = TrainingArguments(
    output_dir="./pubmedbert_crf_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5, # Slightly higher for PubMedBERT
    per_device_train_batch_size=8, # Increased slightly
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=10, # 20 might be overkill with PubMedBERT's efficiency
    fp16=True,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none"
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# ------------------ TRAIN ------------------
print("\n🚀 PubMedBERT + CRF Training Started...\n")
trainer.train()

# ------------------ SAVE ------------------
trainer.save_model("./pubmedbert_crf_model")
tokenizer.save_pretrained("./pubmedbert_crf_model")
print("\n✅ PubMedBERT + CRF Training Completed!")