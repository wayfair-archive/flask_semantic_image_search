import numpy as np

from fixtures import single_ball_tree


def test_single_ball_tree(single_ball_tree):
    assert single_ball_tree.query(np.array([1]), 1) == [{'distance': 1.0, 'metadata': {'item_id': '0'}}]
