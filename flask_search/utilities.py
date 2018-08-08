import base64
import re
import time
from contextlib import contextmanager
from functools import wraps
from io import StringIO

import numpy as np
from flask import current_app as APP
from PIL import Image

RE_IMAGE_HEADER = re.compile(r'^data:image/.+;base64,')


@contextmanager
def record_time_context(name):
    """Context to record milliseconds elapsed for a particular task

    Args:
        name (str): Name to log to statsd
    """
    start_time = time.time()
    yield
    total_time = 1000 * (time.time() - start_time)
    APP.logger.debug('record_time {}: {}ms'.format(name, round(total_time, 3)))


def record_time(func):
    """Decorator for logging a function's runtime.
    It automatically assigns the stat a name based on its location in the flask app
    and function name. It gets the module name from the file where the function is located,
    then appends the function name to the module name.
    """
    index = func.__module__.index('.') + 1
    prefix = func.__module__[index:]
    stats_name = '{}.{}'.format(prefix, func.func_name)

    @wraps(func)
    def wrapped(*args, **kwargs):
        with record_time_context(stats_name):
            result = func(*args, **kwargs)
        return result
    return wrapped


def load_image_str(base64_str):
    """Load an image from a base64 string.
    Pillow requires the the headers be removed from the string to load it properly

    Args:
        base64_str (str): Base-64 encoded image file

    Returns:
        PIL.Image
    """
    image_data = base64.b64decode(RE_IMAGE_HEADER.sub('', base64_str))
    return Image.open(StringIO(image_data))


def pad_image_to_square(image):
    """Takes an image and pads it with whitespace to make it a square

    Args:
        image (PIL.Image): Pillow image

    Returns:
        PIL.Image
    """
    width, height = image.size
    if width != height:
        max_dimension = max(image.size)
        square_image = Image.new('RGB', (max_dimension, max_dimension), 'white')
        paste_location = ((max_dimension - width) // 2, (max_dimension - height) // 2)
        square_image.paste(image, paste_location)
        return square_image
    else:
        return image


def preprocess_image_keras_inception(image, size=(299, 299)):
    """Takes an image, converts it to RGB, pad to a square image, resizes, and normalizes
    pixel values. This is based on how Inception models preprocessed images during training.
    At inference time, the same preprocessing steps should be run.

    Args:
        image (PIL.Image): Image to process
        size (Optional[tuple]): Tuple of width and height for new image

    Returns:
        np.array
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image = pad_image_to_square(image)

    if image.size != size:
        image = image.resize(size, Image.ANTIALIAS)

    X = np.array(image, dtype=np.float32)
    X = np.expand_dims(X, axis=0)
    X /= 127.5
    X -= 1.0

    return X
