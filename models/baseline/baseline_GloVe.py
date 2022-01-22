import numpy as np
from gensim.models import KeyedVectors

from models.baseline import BaselineWord2Vec


class BaselineGloVe(BaselineWord2Vec):
    def __init__(self, filepath: str, path_to_embeddings: str):
        super().__init__(filepath)
        self.path_to_embeddings = path_to_embeddings

    def vectorize(self):
        """
        This method computes the vector forms of each token found in the datasets info.

        :return: None
        """

        self.vectorizer = KeyedVectors.load_word2vec_format(self.path_to_embeddings)

        self.embedding = [
            [
                np.mean(
                    [
                        self.vectorizer[word]
                        if word in self.vectorizer
                        else self.vectorizer["unk"]
                        for word in dataset
                    ]
                )
            ]
            for dataset in self.transformed_data
        ]
