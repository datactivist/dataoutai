import re
import json
from typing import Dict

import numpy as np
import sklearn

from gensim.utils import tokenize
from gensim.models import Word2Vec, KeyedVectors
from sklearn.cluster import AgglomerativeClustering

from models.models_tools import filter_data


class BaselineWord2Vec:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.data = []
        self.transformed_data: list = []
        self.transformed_data_ids: list = []
        self.vectorizer = None
        self.embedding = []

    def load_json(self) -> None:
        """
        This method loads data from the filepath attribute.

        :return: None
        """
        with open(self.filepath, encoding="utf8") as f:
            data = json.load(f)["datasets"]
            for dataset in data:
                self.data.append(dataset)

    def load_and_prepare(
        self,
        filepath: str = "",
        tags_filters=None,
        random_data: int = None,
    ):
        """
        Load the data from the given filepath if not an empty string, else from the filepath attribute.
        Builds the corpus of texts and creates the Word2Vec vectorizer.


        :param random_data:
        :param filepath: an optional parameter, a string indicating from where the data must be loaded
        :param tags_filters: List of tags to include in the tf_idf representation default to
        ["dataset_name", "keywords", "description"]
        :param random_data: Number of random data picked from the transformed datas
        :return: None
        """
        if tags_filters is None:
            tags_filters = ["dataset_name", "keywords", "description"]

        if filepath != "":
            self.filepath = filepath

        self.load_json()
        filtered_data = filter_data(self.filepath, tags_filters, random_data)
        self.transformed_data = [
            list(tokenize(dataset, deacc=True, lowercase=True))
            for dataset in list(filtered_data.values())
        ]
        self.transformed_data_ids = list(filtered_data.keys())
        self.vectorize()

    def kmean_clustering(
        self,
        clustering_model: sklearn.cluster = AgglomerativeClustering(n_clusters=10),
    ) -> Dict:
        """
        Compute cluster given a skelarn clustering model
        :param clustering_model: A cluster model from sklearn.cluster
        :return: A dict of clustered datas from the actual embedding
        """
        clustering_model.fit(self.embedding)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[self.transformed_data_ids[sentence_id]] = cluster_id

        return clustered_sentences

    def build_corpus_from_data(self):
        """
        This method organizes all the text contained in the loaded datasets info in a single list of strings, except for
        the dataset description. Each string represents one unique dataset.

        :return: None
        """
        self.words = []
        for dataset in self.data:
            dataset_as_string = ""
            dataset_as_string += " ".join(dataset["metadata"]["keywords"])
            dataset_as_string += dataset["author"]
            dataset_as_string += dataset["licence"]
            dataset_as_string += dataset["geographic_hold"]
            tokens = re.split(r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b", dataset_as_string)
            self.words.append(tokens)

    def vectorize(self):
        """
        This method computes the vector forms of each token found in the datasets info.

        :return: None
        """

        self.vectorizer = Word2Vec(
            self.transformed_data, min_count=1, vector_size=100, window=10
        )
        self.embedding = [
            [np.mean([self.vectorizer.wv[word] for word in dataset])]
            for dataset in self.transformed_data
        ]

    def get_k_nearest(self, dataset_index: int, k: int = 5, print_result: bool = True):
        """
        This method computes and returns the names of the k-nearest neighbors of the provided dataset with respect to
        the cosine similarity.

        :param dataset_index: an integer, the index of the dataset from which to compute the similarities
        :param k: an integer, the number of "near" datasets to return
        :param print_result: a boolean, indicates whether to print out the result or not

        :return: an array containing the names of the k nearest datasets from the given dataset
        """
        similarities = []
        target_dataset = self.embedding[dataset_index]

        a = np.linalg.norm(target_dataset)

        for index, dataset in enumerate(self.embedding):
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
