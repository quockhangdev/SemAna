import sys
sys.path.append(".")
from utils.dataset_loader import load_imdb_dataset, remove_html_tags

dataset = load_imdb_dataset()

def clean_text(example):
    example["text"] = remove_html_tags(example["text"])
    return example

cleaned_dataset = dataset.map(clean_text)

if __name__ == "__main__":
    print(cleaned_dataset["train"][0]["text"])