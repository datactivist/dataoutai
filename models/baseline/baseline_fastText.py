import fasttext.util
import numpy as np
from models.baseline import BaselineWord2Vec


class BaselineFastText(BaselineWord2Vec):
    def __init__(self, filepath: str, path_to_embeddings: str = None):
        super().__init__(filepath, path_to_embeddings)
        self.path_to_embeddings = path_to_embeddings

    def vectorize(self):
        """
        This method computes the vector forms of each token found in the datasets info.

        :return: None
        """
        if not self.path_to_embeddings:
            fasttext.util.download_model("fr", if_exists="ignore")
            self.vectorizer = fasttext.load_model("cc.fr.300.bin")
        else:
            self.vectorizer = fasttext.load_model(self.path_to_embeddings)

        self.embedding = [
            [np.mean([self.vectorizer.get_word_vector(word) for word in dataset])]
            for dataset in self.transformed_data
        ]
