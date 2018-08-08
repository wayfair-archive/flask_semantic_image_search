from flask_search.settings.base import BaseConfig


class TestingConfig(BaseConfig):
    """Testing configurations"""

    TESTING = True

    WTF_CSRF_ENABLED = False

    # Static Files
    SEARCH_INDEX_FILE = 'data/search_index.pkl'
