from flask_search.app import warm_tensorflow


def post_fork(server, worker):
    """Run this function after the master process is forked.
    Warm the Tensorflow package here so every worker can quickly
    run Tensorflow functions."""
    warm_tensorflow()
