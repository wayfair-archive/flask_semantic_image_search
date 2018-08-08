import os
import logging
import logging.handlers

import wtforms_json
import numpy as np
import tensorflow as tf
import tensorflow.contrib
from flask import Flask

from flask_search.common.responses import page_not_found, healthy
from flask_search.common.models import BallTreeEmbeddingIndex
from flask_search.search.views import search_blueprint


__all__ = ['create_app']

DEFAULT_BLUEPRINTS = (
    search_blueprint,
)


def create_app(config=None):
    app = Flask(__name__)
    configure_app(app, config)
    configure_logging(app)
    configure_blueprints(app, DEFAULT_BLUEPRINTS)
    configure_extensions(app)
    configure_errorhandlers(app)
    load_search_index(app)
    warm_tensorflow()
    app.logger.info('Flask Semantic Image Search powering up on PID {}'.format(os.getpid()))

    return app


def configure_app(app, config):
    """Configure app settings from a python object.
    """
    if config:
        config_class = 'flask_search.settings.{}.{}Config'.format(config.lower(), config.capitalize())
        app.config.from_object(config_class)
    elif 'FLASK_ENV' in os.environ:
        environ = os.environ['FLASK_ENV']
        config_class = 'flask_search.settings.{}.{}Config'.format(environ.lower(), environ.capitalize())
        app.config.from_object(config_class)
    else:
        raise ValueError('No FLASK_ENV environment variable. Cannot load app configuration.')


def configure_logging(app):
    """Configure logger settings. Assign logger to root level package to access outside app context.
    Add a file handler to the root level logger"""
    if app.debug or app.testing:
        return

    root_logger = logging.getLogger('flask_search')
    root_logger.setLevel(app.config.get('LOG_LEVEL', logging.WARNING))

    formatter = logging.Formatter(app.config.get('LOGGING_FORMAT_STR'),
                                  app.config.get('LOGGING_FORMAT_DATE'))
    file_handler = logging.handlers.RotatingFileHandler(app.config.get('LOG_FILEPATH', 'flask_search.log'),
                                                        maxBytes=app.config.get('LOG_FILE_MAX_BYTES', 1000000),
                                                        backupCount=app.config.get('LOG_FILE_BACKUPS', 5))
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def configure_blueprints(app, blueprints):
    """Configure blueprints in views. Add a health checkpoint to each for load balancers."""
    for blueprint in blueprints:
        blueprint.add_url_rule('/health/', view_func=healthy)
        app.register_blueprint(blueprint)


def configure_extensions(app):
    """Initialize any extensions."""

    # wtforms_json extends wtforms to use json request bodies
    wtforms_json.init()


def configure_errorhandlers(app):
    """Configure errorhandlers"""
    app.errorhandler(404)(page_not_found)


def warm_tensorflow():
    """Warm Tensorflow packages"""
    # The first call to this function takes a second
    _ = tf.contrib.util.make_tensor_proto(np.random.randn(1, 1, 1, 1))


def load_search_index(app):
    """Load the search indexes to perform kNN searches"""
    if os.environ.get('FLASK_APP') and not os.environ.get('WERKZEUG_RUN_MAIN'):
        app.logger.warn('Skip loading index on Flask debug reloader thread on PID {}'.format(os.getpid()))
        return

    app.logger.info('Loading search index')
    app.search_index = BallTreeEmbeddingIndex(app.config['SEARCH_INDEX_FILE'])
    app.search_index.load()
