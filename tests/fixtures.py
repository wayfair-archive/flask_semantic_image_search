import pytest
from sklearn.neighbors import BallTree

from flask_search.common.models import BallTreeEmbeddingIndex


@pytest.fixture
def single_ball_tree():
    embedding_index = BallTreeEmbeddingIndex('')
    metadata = [{'item_id': '0'}]
    tree = BallTree([[0]])
    embedding_index.populate({'metadata': metadata, 'index': tree})
    return embedding_index
