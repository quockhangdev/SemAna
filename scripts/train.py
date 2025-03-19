import sys
sys.path.append(".")
from transformers import (
    Trainer, TrainingArguments
)
import torch
from preprocess import get_cleaned_dataset
from utils.model_utils import load_base_model
from evaluate import compute_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

raw_dataset = get_cleaned_dataset()
tokenizer, model = load_base_model(device=device)

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
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8, # Edit this value based on your GPU VRAM
    per_device_eval_batch_size=8, # Edit this value based on your GPU VRAM
    num_train_epochs=5, # Edit this value based on your dataset
    weight_decay=0.01,
    learning_rate=5e-5,
    optim="adamw_torch_fused",
    logging_strategy="steps",
    logging_steps=100,
    metric_for_best_model="f1",
    load_best_model_at_end=True,
)

if device.type == "cuda":
    training_args.gpus = 1
    training_args.bf16 = True # Mixed precision training
    # training_args.fp16 = True # Mixed precision training
    # training_args.fp16_opt_level = "O1" # Mixed precision training

print("Training arguments created successfully!", training_args)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

print("Trainer created successfully!")
print("Training started!")

trainer.train()
trainer.save_model("./models/final_model")