import json
import os

from flask_search.utilities import preprocess_image_keras_inception

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


def read_keras_class_json(filename):
    filepath = os.path.join(FILE_DIR, filename)
    with open(filepath) as f:
        data = json.load(f)
    return [info[1] for index, info in sorted(data.items(), key=lambda x: int(x[0]))]


class ModelConfig(object):
    pass


class InceptionResnetV2ModelConfig(ModelConfig):

    name = 'inception_resnet_v2'
    signature = 'prediction'
    timeout = 5
    inputs = ['inputs']
    outputs = ['softmax', 'embedding']
    class_names = read_keras_class_json('imagenet_class_index.json')
    fn_preprocess = staticmethod(preprocess_image_keras_inception)
