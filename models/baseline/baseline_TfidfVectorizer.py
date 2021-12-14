import json
from typing import Tuple, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class TfidfBaseline:
    def __init__(self, file_path: str):

        self.file_path = file_path
        self.tfidf_matrix = None
        self.transformed_data = []

    def change_file_path(self, file_path: str):
        """
        Change the dataset used for the tf-idf similarity
        :param file_path: The path of the dataset to get
        """
        self.file_path = file_path

    def get_similarity(
        self, ind: int = 42, max_docs: int = 5, verbose: bool = True
    ) -> Tuple[list, list]:
        """
        Get the @max_docs most similar document to the docs number @ind
        :param verbose: Choose to print or not the result
        :param ind: Index of the document from which to calculate the similarity
        :param max_docs: Max numbers of similar document returned
        :return Tupple of : indexes of most similar documents, cosine similarity of said documents
        """
        cosine_similarities = linear_kernel(
            self.tfidf_matrix[ind : ind + 1], self.tfidf_matrix
        ).flatten()
        related_docs_indices = cosine_similarities.argsort()[: -max_docs - 2 : -1]
        if verbose:
            print(f'Document "tag" d\'origine :\n{self.transformed_data[ind]}')
            print("")

            for ind in related_docs_indices[1:]:
                print(f"Similarité cosinus de {cosine_similarities[ind]} : ")
                print(self.transformed_data[ind])

        return related_docs_indices, cosine_similarities[related_docs_indices]

    def compute_tfidf(self, max_features: int):
        """
        Compute the similarity matrix of the tf_idf method on previously given dataset (can be changed using .change_file_path(file_path: str) )
        :param max_features: max_features of the TfidfVectorizer
        """
        v = TfidfVectorizer(
            input="content",
            encoding="utf-8",
            decode_error="replace",
            strip_accents="unicode",
            lowercase=True,
            analyzer="word",
            # stop_words=fr_stop,
            token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b",
            ngram_range=(1, 2),
            max_features=max_features,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True,
        )

        with open(self.file_path, encoding="utf8") as f:
            datas = json.load(f)

        datasets = datas["datasets"]
        for d in datasets:
            keyword = ""
            if d["metadata"]["keywords"] is not None:
                keyword = " ".join(d["metadata"]["keywords"])
            description = ""
            if d["metadata"]["description"] is not None:
                description = d["metadata"]["description"]
            author = ""
            if d["author"] is not None:
                author = d["author"]
            licence = ""
            if d["licence"] is not None:
                licence = d["licence"]
            geographic_hold = ""
            if d["geographic_hold"] is not None:
                geographic_hold = " ".join(["geographic_hold"])

            self.transformed_data.append(
                " ".join(
                    [
                        d["dataset_name"],
                        author,
                        licence,
                        geographic_hold,
                        description,
                        keyword,
                    ]
                )
            )

        self.tfidf_matrix = v.fit_transform(self.transformed_data)


# tfidf = TfidfBaseline("../../data/datasud.json")
# tfidf.compute_tfidf(20000)
# tfidf.get_similarity(42)
