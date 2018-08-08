import time

import numpy as np

from flask_search.common.daos import TensorflowPredictRequest


class TensorflowImageEmbeddingRequest(TensorflowPredictRequest):

    """An image embedding request to Tensorflow Serving
    """

    def set_image(self, image_array):
        self.set_input('input', image_array)

    def get_outputs(self):
        """Get the embedding and classification outputs of from Tensorflow Serving"""
        response = self.stub_predict()
        if response and response.outputs:
            embedding = np.array(response.outputs['embedding'].float_val, dtype=np.float32)
            if 'softmax' in response.outputs:
                softmax = np.array(response.outputs['softmax'].float_val, dtype=np.float32)
            else:
                softmax = None

            return {
                'embedding': embedding,
                'softmax': softmax
            }
        else:
            return {}
