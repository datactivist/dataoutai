from typing import Tuple, List, Dict, Optional

import numpy as np
import sklearn
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD

from models.baseline import TfidfBasedBaseline


class LSABaseline(TfidfBasedBaseline):
    def __init__(self, file_path: str = ""):
        super().__init__(file_path)
        self.lsa_matrix: Optional[np.ndarray] = None

    def compute_lsa(self):
        self.lsa_matrix = TruncatedSVD(
            n_components=50, n_iter=5, random_state=42
        ).fit_transform(self.tfidf_matrix)

    def get_similarity(
        self, ind: int = 42, max_docs: int = 5, verbose: bool = True
    ) -> Tuple[list, list]:
        if self.lsa_matrix is None:
            self.compute_lsa()
        return self.compute_similarity(self.lsa_matrix, ind, max_docs, verbose)

    def kmean_clustering(
        self, clustering_model: sklearn.cluster = AgglomerativeClustering(n_clusters=10)
    ) -> Dict:
        if self.lsa_matrix is None:
            self.compute_lsa()
        clustering_model.fit(self.lsa_matrix)
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[self.transformed_data_ids[sentence_id]] = cluster_id

        return clustered_sentences

    def compute(self, max_features: int, tags_filters: List[str] = None):
        super().compute(max_features, tags_filters)
        self.compute_lsa()


if __name__ == "__main__":
    import pickle
    from sklearn.cluster import KMeans

    lsa = LSABaseline("../../data/datasud.json")

    lsa.compute(500)
    clusters = lsa.kmean_clustering(KMeans(n_clusters=15))
    print(clusters)
    with open(
        r"../../models/evaluation/clusters/lsa_clusters_datasud_km_15.pkl", "wb"
    ) as output_file:
        pickle.dump(clusters, output_file)
