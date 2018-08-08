import pickle

from sklearn.neighbors import BallTree


class EmbeddingIndex(object):

    def query(self, X, k):
        raise NotImplementedError()


class BallTreeEmbeddingIndex(EmbeddingIndex):

    """A fast kNN search structure built with sklearn's BallTree as a backend.

    Attributes:
        data_file (str): Path to pickle file with data to populate the object
        metadata (List[dict]): List of metadata for each retrieval image,
                               order by ID defined by the BallTree
        tree (BallTree): kNN search structure from sklearn with all image embeddings
    """

    def __init__(self, data_file):
        self.data_file = data_file

    def populate(self, data):
        self.metadata = data['metadata']
        self.tree = data['index']

    def load(self):
        data = self.get_data_from_disk()
        self.populate(data)

    def get_data_from_disk(self):
        with open(self.data_file, 'rb') as f:
            return pickle.load(f)

    def query(self, X, k):
        """Get the top k nearest neighbors for some input vector X

        Args:
            X (ndarray): Image embedding numpy array of shape (D,) where D is number of dimensions
            k (int): Number of results to return

        Returns:
            List[dict]
        """
        X = X.reshape(1, -1)
        distances, indexes = self.tree.query(X, k)
        results = []
        for index, dist in zip(indexes[0], distances[0]):
            result = {
                'distance': dist,
                'metadata': self.metadata[index]
            }
            results.append(result)
        return results
