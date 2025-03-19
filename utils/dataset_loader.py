import datasets

__all__ = ["load_imdb_dataset"]

def load_imdb_dataset():
    ds = datasets.load_dataset("stanfordnlp/imdb")
    return ds

if __name__ == "__main__":
    ds = load_imdb_dataset()
    print(ds["train"][0]) # {'label': 0, 'text': 'For a movie that gets..'}