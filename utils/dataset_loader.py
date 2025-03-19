import datasets
from bs4 import BeautifulSoup

__all__ = ["load_imdb_dataset", "remove_html_tags"]

def load_imdb_dataset():
    ds = datasets.load_dataset("stanfordnlp/imdb")
    return ds

def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

if __name__ == "__main__":
    ds = load_imdb_dataset()
    print(ds)
    print(ds["train"][0]) # {'label': 0, 'text': 'For a movie that gets..'}