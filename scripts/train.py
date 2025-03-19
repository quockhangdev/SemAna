import sys
sys.path.append(".")
from transformers import (
    Trainer, TrainingArguments
)
import torch
from preprocess import get_cleaned_dataset
from utils.model_utils import load_base_model
from evaluate import compute_metrics

raw_dataset = get_cleaned_dataset()
tokenizer, model = load_base_model()

print("Model loaded successfully!")
print(model)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, return_tensors="pt")

raw_dataset =  raw_dataset.rename_column("label", "labels")
tokenized_dataset = raw_dataset.map(tokenize_function, batched=True,remove_columns=["text"])

print(tokenized_dataset["train"].features.keys())
print("Tokenized dataset created successfully!")

training_args = TrainingArguments(
    output_dir="./models/checkpoints",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    learning_rate=5e-5,
    optim="adamw_torch_fused",
    logging_strategy="steps",
    logging_steps=100,
    metric_for_best_model="f1",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("./models/final_model")