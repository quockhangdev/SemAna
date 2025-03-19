import numpy as np
from sklearn.metrics import f1_score, accuracy_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    f1 = f1_score(
            labels, predictions, labels=labels, pos_label=1, average="weighted"
        )
    acc = accuracy_score(labels, predictions)
    return {"f1": float(f1) if f1 == 1 else f1, "accuracy": acc}