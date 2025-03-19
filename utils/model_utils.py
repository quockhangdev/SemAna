from transformers import (
    AutoTokenizer, ModernBertForSequenceClassification
)
import torch

def load_base_model(
    num_labels=2, label2id={"neg": 0, "pos": 1}, id2label={0: "neg", 1: "pos"}
):
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    tokenizer.model_max_length = 512

    model = ModernBertForSequenceClassification.from_pretrained(
        "answerdotai/ModernBERT-base",
        num_labels=num_labels, label2id=label2id, id2label=id2label,
    )

    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = load_base_model()
    print("Model loaded successfully!")
    print(model)

    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    predicted_class_id = logits.argmax().item()
    predicted_class = model.config.id2label[predicted_class_id]

    print(f"Predicted class: {predicted_class} ({predicted_class_id})")