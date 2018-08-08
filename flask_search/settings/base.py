import os


class BaseConfig(object):
    """Configuration common to all environments"""

    # Logging
    LOGGING_FORMAT_STR = '%(asctime)s [%(process)d] [%(levelname)s] %(name)-16s %(message)s'
    LOGGING_FORMAT_DATE = '%Y-%m-%d %H:%M:%S'

    # Tensorflow Serving
    TENSORFLOW_SERVING_HOSTPORT = 'tensorflow_serving:9091'
