from typing import Tuple, Dict

import sklearn
from sklearn.cluster import AgglomerativeClustering

from models.baseline import TfidfBasedBaseline


class TfidfBaseline(TfidfBasedBaseline):
    def __init__(self, file_path: str = ""):
        super().__init__(file_path)

    def get_similarity(
        self, ind: int = 42, max_docs: int = 5, verbose: bool = True
    ) -> Tuple[list, list]:
        return self.compute_similarity(self.tfidf_matrix, ind, max_docs, verbose)

    def kmean_clustering(
        self, clustering_model: sklearn.cluster = AgglomerativeClustering(n_clusters=10)
    ) -> Dict:
        """
        Compute cluster given a sklearn clustering model
        :param clustering_model: A cluster model from sklearn.cluster
        :return: A dict of clustered datas from the actual embedding
        """
        clustering_model.fit(self.tfidf_matrix.toarray())
        cluster_assignment = clustering_model.labels_

        clustered_sentences = {}
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_sentences[self.transformed_data_ids[sentence_id]] = cluster_id

        return clustered_sentences


if __name__ == "__main__":
    import pickle
    from sklearn.cluster import KMeans

    tfidf = TfidfBaseline("../../data/datasud.json")

    tfidf.compute(500)
    clusters = tfidf.kmean_clustering(KMeans(n_clusters=15))
    print(clusters)
    with open(
        r"../../models/evaluation/clusters/tfidf_clusters_datasud_km_15.pkl", "wb"
    ) as output_file:
        pickle.dump(clusters, output_file)
