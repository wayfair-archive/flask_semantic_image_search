import tensorflow as tf
import tensorflow.contrib
from flask import current_app as APP
from grpc.framework.interfaces.face import face
from tensorflow_serving.apis import predict_pb2

from flask_search.utilities import record_time_context


class TensorflowPredictRequest(object):

    """Contains a prediction request for Tensorflow Serving

    Attributes:
        stub (grpc.beta._client_adaptations._DynamicStub): Stub connecting to the TF Serving server
        config (flask_search.model_zoo.configs.ModelConfig): Model config class with metadata
        request (predict_pb2.PredictRequest): Predict request API object
        timeout (Optional[float]): Override default timeout seconds for requests
    """

    def __init__(self, stub, config, timeout=None):
        self.stub = stub
        self.config = config
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = config.name
        self.request.model_spec.signature_name = config.signature
        self.timeout = timeout or config.timeout

    def set_input(self, name, value):
        """Set an input the the Tensorflow model

        Args:
            name (str): Input layer name
            value (object): Input value to layer
        """
        self.request.inputs[name].CopyFrom(
            tf.contrib.util.make_tensor_proto(value)
        )

    def stub_predict(self):
        """Send the result to Tensorflow Serving and get the output"""
        try:
            stats_name = 'serving.{}.stub_predict'.format(self.__class__.__name__)
            with record_time_context(stats_name):
                return self.stub.Predict(self.request, self.timeout)
        except face.ExpirationError as e:
            APP.logger.error('Tensorflow Serving {} timeout'.format(self.config.name))
        except face.AbortionError as e:
            APP.logger.error('Tensorflow Serving {} not reachable'.format(self.config.name))
