import sys
sys.path.append(".")
from utils.dataset_loader import load_imdb_dataset, remove_html_tags

__all__ = ["get_cleaned_dataset"]

dataset = load_imdb_dataset()

def clean_text(example):
    example["text"] = remove_html_tags(example["text"])
    return example

cleaned_dataset = dataset.map(clean_text)

def get_cleaned_dataset():
    return cleaned_dataset

if __name__ == "__main__":
    print(cleaned_dataset["train"][0]["text"])