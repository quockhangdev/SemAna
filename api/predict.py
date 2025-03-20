import torch

__all__ = ["predict"]

def predict(model, tokenizer, text: str):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    predicted_class = model.config.id2label[predicted_class_id]
    return predicted_class

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from utils.model_utils import load_model
    text = "I love this movie!"
    tokenizer, model = load_model()
    prediction = predict(model, tokenizer, text)
    print(prediction)