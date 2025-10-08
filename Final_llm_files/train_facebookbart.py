import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json
import random

# 1. Load and prepare dataset
def load_and_split_data(jsonl_path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * 0.9)

    train_data = data[:split_idx]
    eval_data = data[split_idx:]

    train_dataset = Dataset.from_dict({
        "input": [d["question"] for d in train_data],
        "labels": [d["answer"] for d in train_data]
    })
    eval_dataset = Dataset.from_dict({
        "input": [d["question"] for d in eval_data],
        "labels": [d["answer"] for d in eval_data]
    })

    return train_dataset, eval_dataset

# ✅ Load fullqa_trimmed.jsonl
train_dataset, eval_dataset = load_and_split_data("fullqa_trimmed.jsonl")

# 2. Model and tokenizer
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# 3. Preprocess
def preprocess_function(examples):
    model_inputs = tokenizer(examples["input"], padding=True, truncation=True, max_length=128)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["labels"], padding=True, truncation=True, max_length=128)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_eval = eval_dataset.map(preprocess_function, batched=True)

# 4. Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 5. Training arguments
training_args = TrainingArguments(
    output_dir="./results_fullqa",
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    warmup_steps=100,
    weight_decay=0.01,
    logging_dir="./logs_fullqa",
    logging_steps=10,
    eval_strategy="steps",  # ✅ Add this
    save_strategy="steps",        # ✅ Match this
    save_steps=500,
    eval_steps=500,               # ✅ Optional but recommended
    load_best_model_at_end=True,
    learning_rate=1e-4,
    fp16=True if torch.cuda.is_available() else False
)



# 6. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

# 7. Train
trainer.train()

# 8. Save
trainer.save_model("./trained_model_fullqa")
tokenizer.save_pretrained("./trained_model_fullqa")
