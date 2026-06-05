import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    BertModel
)
import torch
import torch.nn as nn
from TorchCRF import CRF

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

        fixed_tags = []

        for tag in ner_tags:

            if isinstance(tag, int):
                tag = id2label[tag]

            if "MICROBE" in tag or tag not in label2id:
                fixed_tags.append("O")
            else:
                fixed_tags.append(tag)

        if len(tokens) == len(fixed_tags):
            clean_data.append({
                "tokens": tokens,
                "ner_tags": fixed_tags
            })

    return clean_data


merged_data = normalize_dataset(merged_data)

dataset = Dataset.from_list(merged_data).train_test_split(
    test_size=0.2,
    seed=42
)

# ---------------- TOKENIZER ----------------

model_checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_and_align(examples):

    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        padding="max_length",
        truncation=True,
        max_length=256
    )

    labels = []

    for i, ner_tags in enumerate(examples["ner_tags"]):

        word_ids = tokenized.word_ids(batch_index=i)

        prev_word = None
        label_ids = []

        for word_id in word_ids:

            if word_id is None:
                label_ids.append(-100)

            elif word_id != prev_word:
                label_ids.append(label2id[ner_tags[word_id]])

            else:
                label_ids.append(-100)

            prev_word = word_id

        labels.append(label_ids)

    tokenized["labels"] = labels

    return tokenized


tokenized_ds = dataset.map(
    tokenize_and_align,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# ---------------- MODEL ----------------

class PubMedBERT_CRF(nn.Module):

    def __init__(self, model_name, num_labels):
        super().__init__()

        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                token_type_ids=None):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        sequence_output = self.dropout(outputs.last_hidden_state)

        emissions = self.classifier(sequence_output)

        mask = attention_mask.bool()

        if labels is not None:

            labels_clean = labels.clone()
            labels_clean[labels_clean == -100] = 0

            log_likelihood = self.crf(emissions, labels_clean, mask=mask)

            loss = -log_likelihood.mean()

            return {"loss": loss}

        else:
            decoded = self.crf.decode(emissions, mask=mask)

            max_len = input_ids.size(1)

            padded = [seq + [0] * (max_len - len(seq)) for seq in decoded]

            return {"logits": torch.tensor(padded).to(input_ids.device)}


model = PubMedBERT_CRF(model_checkpoint, num_labels).to(device)

# ---------------- TRAINING CONFIG ----------------

training_args = TrainingArguments(

    output_dir="./pubmedbert_crf_model",

    evaluation_strategy="no",   # 🔥 CRITICAL FIX
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # effective batch = 16

    num_train_epochs=10,

    fp16=True,

    warmup_ratio=0.1,
    weight_decay=0.01,

    logging_steps=50,

    save_total_limit=2,

    remove_unused_columns=False,

    report_to="none"
)

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# ---------------- TRAIN ----------------

print("\n🚀 PubMedBERT + CRF Training Started\n")

trainer.train()

# ---------------- SAVE FINAL MODEL ----------------

trainer.save_model("./pubmedbert_crf_model")
tokenizer.save_pretrained("./pubmedbert_crf_model")

print("\n✅ Training Finished Successfully!")
