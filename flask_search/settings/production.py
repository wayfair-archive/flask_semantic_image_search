import logging

from flask_search.settings.base import BaseConfig


class ProductionConfig(BaseConfig):
    """Production configurations"""

    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FILEPATH = 'flask_search.log'
    LOG_FILE_MAX_BYTES = 1000000  # 1 MB
    LOG_FILE_BACKUPS = 5

    WTF_CSRF_ENABLED = False

    # Static Files
    SEARCH_INDEX_FILE = 'data/search_index.pkl'
