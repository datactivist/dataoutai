from typing import List, Any, Union, Dict, Tuple
from numpy import ndarray
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
from torch import Tensor
from models.models_tools import filter_data
from sklearn.cluster import AgglomerativeClustering
import sklearn.cluster


class ContextualEmbeddingModel:
    def __init__(self, file_path: str = "", model: SentenceTransformer = None):
        self.device = None
        self.file_path: str = file_path
        self.configuration: List[str] = []
        self.model: SentenceTransformer = model
        self.transformed_data_ids: List[str] = []
        self.transformed_data: List[str] = []
        self.embedding: Union[List[Tensor], ndarray, Tensor] = []

    def set_file_path(self, file_path: str):
        """
        :param file_path: Set the current datasets file used for the embedding !reset the others parameters!
        """
        self.file_path = file_path

    def set_model(self, model: str):
        """
        :param model: Set the current model used for the embedding !reset the others parameters!
        """
        self.model = model

    def compute_embedding(
        self,
        tags_filters=None,
        convert_to_tensor: bool = True,
        convert_to_numpy: bool = False,
        show_progress_bar: bool = True,
        device: str = "cpu",
        random_data: int = None,
    ) -> Any:
        """
        :param device: Which torch.device to use for the computation (from SentenceTransformer)
        :param show_progress_bar: Output a progress bar when encode sentences (from SentenceTransformer)
        :param convert_to_numpy: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy (from SentenceTransformer)
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy (from SentenceTransformer)
        :param tags_filters: List of tags to include in the tf_idf representation default to
        ["dataset_name", "keywords", "description"]
        :param random_data: Number of random data picked from the transformed datas
        """
        self.device = device

        self.configuration = tags_filters
        filtered_data = filter_data(self.file_path, tags_filters, random_data)
        self.transformed_data = list(filtered_data.values())
        self.transformed_data_ids = list(filtered_data.keys())
        self.embedding = self.model.encode(
            self.transformed_data,
            show_progress_bar=show_progress_bar,
            convert_to_tensor=convert_to_tensor,
            convert_to_numpy=convert_to_numpy,
            device=device,
        )

    def save_embedding(self, name: str):
        """
        Save the embedding as a .pkl
        :param name:
        :return:
        """
        with open(f"{name}.pkl", "wb") as fo:
            pickle.dump(
                {
                    "transformed_data": self.transformed_data,
                    "transformed_data_ids": self.transformed_data_ids,
                    "embeddings": self.embedding,
                },
                fo,
            )

    def load_embedding(self, name: str):
        """
        Load a .pkl embedding
        :param name:
        :return:
        """
        with open(f"{name}.pkl", "rb") as fi:
            pickle_file = pickle.load(fi)
            self.transformed_data = pickle_file["transformed_data"]
            self.transformed_data_ids = pickle_file["transformed_data_ids"]
            self.embedding = pickle_file["embeddings"]

    def kmean_clustering(
        self,
        clustering_model: sklearn.cluster = AgglomerativeClustering(n_clusters=10),
    ) -> Dict:
        """
        Compute cluster given a skelarn clustering model
        :param clustering_model: A cluster model from sklearn.cluster
        :return: A dict of clustered datas from the actual embedding
        """
        normalized_embedding = self.embedding / torch.linalg.norm(
            self.embedding, keepdims=True
        )
        normalized_embedding = normalized_embedding.cpu().numpy()
        clustering_model.fit(normalized_embedding)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[self.transformed_data_ids[sentence_id]] = cluster_id

        return clustered_sentences

    def cosine_similarity(self, query: str, top_k: int = 5) -> Tuple[Any, Any]:
        """
        Compute the cosine similarity for a given query and return the @top_k higher similarities
        :param query: Query
        :param top_k: Number of document to return
        :return: Lists of scores and ids of the @top_k higher similarities
        """
        top_k = min(top_k, len(self.transformed_data))
        query_embedding = self.model.encode(
            query, convert_to_tensor=True, device=self.device
        )
        cos_scores = util.cos_sim(query_embedding, self.embedding)[0]

        top_results = torch.topk(cos_scores, k=top_k + 1)
        scores = []
        idxs = []
        for score, idx in zip(top_results[0], top_results[1]):
            scores.append(score.item())
            idxs.append(idx.item())

        return scores, idxs
