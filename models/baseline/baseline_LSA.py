import json
import random

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_json(filepath: str) -> dict:
    with open(filepath, encoding="utf8") as f:
        data = json.load(f)
    return data


def build_corpus_from_dataset(json_dump):
    c = []
    for dataset in json_dump["datasets"]:
        if dataset["metadata"]["description"] == "":
            continue
        if dataset["metadata"]["keywords"]:
            keywords = " ".join(dataset["metadata"]["keywords"])
        else:
            keywords = ""
        author = dataset.get("author", "")
        licence = dataset.get("licence", "")
        geographic_hold = dataset.get("geographic_hold", "")
        c.append(
            f"{dataset['dataset_name']} {author if author else ''} {licence if licence else ''} "
            f"{geographic_hold if geographic_hold else ''} {dataset['metadata']['description']} "
            f"{keywords if keywords else ''}"
        )
    return c


def train_test_split(data, test_size=5):
    random.seed(42)
    test_data = random.sample(data, test_size)
    train_data = []
    for text in corpus:
        if text in test_data:
            continue
        train_data.append(text)
    return train_data, test_data


def find_closest_match(
    dataset, tfidf_transformer, lsa_transformer, lsa_transforms, text
):
    vectorized_test = tfidf_transformer.transform([text])
    lsa_test = lsa_transformer.transform(vectorized_test)
    results = cosine_similarity(lsa_transforms, lsa_test)
    results = results.reshape(len(results))
    index = np.argsort(results)
    print(results[index[-1]])
    return dataset["datasets"][index[-1]]


if __name__ == "__main__":
    datasets = load_json("../../data/datasud.json")
    open_truc = load_json("../../data/opendatasoft.json")
    datasets["count"] += open_truc["count"]
    datasets["datasets"] += open_truc["datasets"]
    # data_gouv = load_json("../../data/datagouv.json")
    # datasets["count"] += data_gouv["count"]
    # datasets["datasets"] += data_gouv["datasets"]

    corpus = build_corpus_from_dataset(datasets)

    train_sample, test_sample = train_test_split(corpus, test_size=10)

    vectorizer = TfidfVectorizer(max_df=0.1)
    X = vectorizer.fit_transform(train_sample)

    lsa = TruncatedSVD(n_components=50, n_iter=5, random_state=42)
    lsa_t = lsa.fit_transform(X)

    for test in test_sample:
        print(test)
        print(find_closest_match(datasets, vectorizer, lsa, lsa_t, test))
        print()
