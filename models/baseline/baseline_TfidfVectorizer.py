import json
from typing import Tuple, List, Dict

import numpy
import sklearn
from scipy.sparse import csr_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from models.models_tools import filter_data
from pydantic import BaseModel
import joblib


class BaseConfiguration(BaseModel):
    name: str = ""
    source: str = ""
    configuration: List[str] = []
    vectorizer_path: str = ""


class TfidfBaseline:
    def __init__(self, file_path: str = ""):

        self.file_path: str = file_path
        self.tfidf_matrix: csr_matrix = None
        self.lsa_matrix: numpy.ndarray = None
        self.transformed_data: list = []
        self.transformed_data_ids: list = []
        self.configuration: List[str] = []
        self.tfidf_vectorizer: TfidfVectorizer = None

    def save_configuration(self, name: str):
        """
        Save a configuration to .config file
        :parameter name: The path of the config file
        """
        config = BaseConfiguration()
        config.name = name
        config.source = self.file_path
        config.configuration = self.configuration
        joblib.dump(self.tfidf_vectorizer, f"{name}_model.joblib")
        config.vectorizer_path = f"{name}_model.joblib"
        with open(f"{name}_config.json", "x") as fo:
            json.dump(json.loads(config.json()), fo)

    def load_configuration(self, path: str):
        """
        Load a configuration from a .config file
        :parameter path: The path of the config file
        """
        config = BaseConfiguration.parse_file(path, content_type="json")
        print(f"Loading configuration : {config.name}")
        self.configuration = config.configuration
        self.file_path = config.source
        self.tfidf_vectorizer = joblib.load(config.vectorizer_path)
        self.transformed_data = filter_data(self.file_path, self.configuration)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.transformed_data)

    def change_file_path(self, file_path: str):
        """
        Change the dataset used for the tf-idf similarity !will reset the whole model!
        :param file_path: The path of the dataset to get
        """
        self.file_path = file_path
        self.tfidf_matrix = None
        self.transformed_data = None
        self.configuration = None
        self.tfidf_vectorizer = None

    def get_similarity(
        self, ind: int = 42, max_docs: int = 5, verbose: bool = True, lsa: bool = False
    ) -> Tuple[list, list]:
        """
        Get the @max_docs most similar document to the docs number @ind
        :param verbose: Choose to print or not the result
        :param ind: Index of the document from which to calculate the similarity
        :param max_docs: Max numbers of similar document returned
        :return Tuple of : indexes of most similar documents, cosine similarity of said documents
        :param lsa: Choose to compute the similarity with LSA or not
        """
        if lsa:
            if self.lsa_matrix is None:
                self.compute_lsa()
            matrix = self.tfidf_matrix
        else:
            matrix = self.lsa_matrix

        cosine_similarities = linear_kernel(matrix[ind : ind + 1], matrix).flatten()
        related_docs_indices = cosine_similarities.argsort()[: -max_docs - 2 : -1]
        print(related_docs_indices)
        if verbose:
            print(f'Document "tag" d\'origine :\n{self.transformed_data[ind]}')
            print("")

            for ind in related_docs_indices[1:]:
                print(f"Similarité cosinus de {cosine_similarities[ind]} : ")
                print(self.transformed_data[ind])

        return related_docs_indices, cosine_similarities[related_docs_indices]

    def kmean_clustering(
        self,
        clustering_model: sklearn.cluster = AgglomerativeClustering(n_clusters=10),
        lsa: bool = False,
    ) -> Dict:
        """
        Compute cluster given a sklearn clustering model
        :param clustering_model: A cluster model from sklearn.cluster
        :return: A dict of clustered datas from the actual embedding
        :param lsa: Choose to compute the similarity with LSA or not
        """
        if lsa:
            if self.lsa_matrix is None:
                self.compute_lsa()
            clustering_model.fit(self.lsa_matrix)
        else:
            clustering_model.fit(self.tfidf_matrix.toarray())
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[self.transformed_data_ids[sentence_id]] = cluster_id

        return clustered_sentences

    def compute_lsa(self):
        self.lsa_matrix = TruncatedSVD(
            n_components=50, n_iter=5, random_state=42
        ).fit_transform(self.tfidf_matrix)

    def compute_tfidf(
        self, max_features: int, tags_filters: List[str] = None, lsa: bool = False
    ):
        """
        Compute the similarity matrix of the tf_idf method on previously given dataset
        (can be changed using .change_file_path(file_path: str) )
        :param max_features: max_features of the TfidfVectorizer
        :param tags_filters: List of tags to include in the tf_idf representation default to
        ["dataset_name", "keywords", "description"]
        :param lsa: Choose to compute LSA or not
        """
        self.configuration = tags_filters
        self.tfidf_vectorizer = TfidfVectorizer(
            input="content",
            encoding="utf-8",
            decode_error="replace",
            strip_accents="unicode",
            lowercase=True,
            analyzer="word",
            token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b",
            ngram_range=(1, 2),
            max_features=max_features,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
        )

        filtered_data = filter_data(self.file_path, tags_filters)
        self.transformed_data = list(filtered_data.values())
        self.transformed_data_ids = list(filtered_data.keys())
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.transformed_data)

        if lsa:
            self.compute_lsa()
