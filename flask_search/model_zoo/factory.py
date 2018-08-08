from flask_search.model_zoo.configs import InceptionResnetV2ModelConfig


MODELS = {
    'inception_resnet_v2': InceptionResnetV2ModelConfig,
}


def get_model_config(model_name):
    """Get the model configs for any given model_name in Tensorflow Serving"""
    config = MODELS.get(model_name)
    if config:
        return config
    else:
        raise ValueError('Cannot find model config')
