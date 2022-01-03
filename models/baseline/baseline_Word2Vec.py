import re
import json
import numpy as np

from gensim.models import Word2Vec


class BaselineWord2Vec:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = []
        self.words = []
        self.vectorizer = None
        self.transformed_datasets = []

        self.load_and_prepare()

    def load_json(self) -> None:
        """
        This method loads data from the filepath attribute.

        :return: None
        """
        with open(self.filepath, encoding="utf8") as f:
            data = json.load(f)["datasets"]
            for dataset in data:
                self.data.append(dataset)

    def load_and_prepare(self, filepath: str = ""):
        """
        Load the data from the given filepath if not an empty string, else from the filepath attribute.
        Builds the corpus of texts and creates the Word2Vec vectorizer.

        :param filepath: an optional parameter, a string indicating from where the data must be loaded

        :return: None
        """
        if filepath != "":
            self.filepath = filepath

        self.load_json()
        self.build_corpus_from_data()
        self.vectorize()

    def build_corpus_from_data(self):
        """
        This method organizes all the text contained in the loaded datasets info in a single list of strings. Each
        string represents one unique dataset.

        :return: None
        """
        self.words = []
        for dataset in self.data:
            dataset_as_string = ""
            if dataset["metadata"]["description"] == "":
                continue
            dataset_as_string += dataset["metadata"]["description"]
            dataset_as_string += " ".join(dataset["metadata"]["keywords"])
            dataset_as_string += dataset.get("author", "")
            dataset_as_string += dataset.get("licence", "")
            dataset_as_string += dataset.get("geographic_hold", "")
            tokens = re.split(r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b", dataset_as_string)
            self.words.append(tokens)

    def vectorize(self):
        """
        This method computes the vector forms of each token found in the datasets info.

        :return: None
        """
        self.vectorizer = Word2Vec(self.words, min_count=1, vector_size=100, window=10)
        self.transformed_datasets = [
            np.mean([self.vectorizer.wv[word] for word in dataset])
            for dataset in self.words
        ]

    def get_k_nearest(self, dataset_index: int, k: int = 5, print_result: bool = True):
        """

        :param dataset_index: an integer, the index of the dataset from which to compute the similarities
        :param k: an integer, the number of "near" datasets to return
        :param print_result: a boolean, indicates whether to print out the result or not

        :return: an array containing the names of the k nearest datasets from the given dataset
        """
        similarities = []
        target_dataset = self.transformed_datasets[dataset_index]

        a = np.linalg.norm(target_dataset)

        for index, dataset in enumerate(self.transformed_datasets):
            if index != dataset_index:
                b = np.linalg.norm(dataset)
                similarities.append(
                    np.linalg.norm(np.array(target_dataset) - np.array(dataset))
                    / (a * b)
                )

        neighbours_indices = np.argsort(similarities)[-k:]

        if print_result:
            print(
                np.array([dataset["dataset_name"] for dataset in self.data])[
                    neighbours_indices
                ]
            )

        return np.array([dataset["dataset_name"] for dataset in self.data])[
            neighbours_indices
        ]
