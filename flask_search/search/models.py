import numpy as np


class VisualSearchImage(object):

    """An incoming image to find similar images for

    Attributes:
        config (ModelConfig): Config for the model used to embed the image
        embedding (ndarray): Embedding from the CNN model
        softmax (ndarray): Array of classification probabilities
    """

    config = None
    embedding = None
    softmax = None

    def __init__(self, config):
        self.config = config

    def populate(self, data):
        self.embedding = data.get('embedding', self.embedding)
        self.softmax = data.get('softmax', self.softmax)

    def get_classifications(self, k):
        classifications = []
        if self.softmax is not None:
            sorted_class_ids = self.softmax.argsort()[-k:][::-1]
            for i in sorted_class_ids:
                classifications.append({
                    'name': self.config.class_names[i],
                    'score': float(self.softmax[i])
                })
        return classifications


class VisualSearchImageResult(object):

    """A result for an image search

    Attributes:
        distance (float): Distance from the query image in the latent space
        filename (str): Name of the image file for this result
        item_id (str): Identifier for which item this image result represents
    """

    distance = None
    filename = None
    item_id = None

    def populate(self, data):
        self.distance = data.get('distance', self.distance)
        self.filename = data.get('filename', self.filename)
        self.item_id = data.get('item_id', self.item_id)

    @property
    def image_url(self):
        return 'images/' + self.filename

    def to_dict(self):
        return {
            'distance': self.distance,
            'filename': self.filename,
            'item_id': self.item_id,
            'image_url': self.image_url
        }
