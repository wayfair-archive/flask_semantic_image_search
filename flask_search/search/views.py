import numpy as np
from flask import current_app as APP
from flask import Blueprint, jsonify, request, render_template
from PIL import Image

import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

from flask_search.search.forms import SearchForm
from flask_search.search.daos import TensorflowImageEmbeddingRequest
from flask_search.search.models import VisualSearchImage, VisualSearchImageResult
from flask_search.model_zoo.factory import get_model_config
from flask_search.common.responses import invalid_request, server_error

search_blueprint = Blueprint('search', __name__, url_prefix='/search')


@search_blueprint.route('/', methods=['GET'])
def search():
    form = SearchForm()
    return render_template('search/index.html', form=form)


@search_blueprint.route('/image/', methods=['POST'])
def search_image():
    form = SearchForm()
    if form.validate_on_submit():
        config = get_model_config('inception_resnet_v2')

        # Load image from POST and preprocess
        image_file = form.image.data
        image = Image.open(image_file)
        image = config.fn_preprocess(image)
        image_file.close()

        # Create Tensorflow Serving connection
        channel = grpc.insecure_channel(APP.config['TENSORFLOW_SERVING_HOSTPORT'])
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

        # Do Tensorflow Serving result and populate model
        tf_request = TensorflowImageEmbeddingRequest(stub, config)
        tf_request.set_image(image)
        outputs = tf_request.get_outputs()
        if not outputs:
            return server_error('Tensorflow Serving did not respond')
        image_model = VisualSearchImage(config)
        image_model.populate(outputs)

        # Do kNN search and populate models
        results = APP.search_index.query(image_model.embedding, form.rows.data)
        result_models = []
        for r in results:
            result_model = VisualSearchImageResult()
            result_model.populate(r)
            result_model.populate(r['metadata'])
            result_models.append(result_model)

        # Show as json or simple html page
        if form.format.data == 'json':
            return jsonify(
                results=[r.to_dict() for r in result_models],
                classifications=image_model.get_classifications(10)
            )
        elif form.format.data == 'html':
            return render_template('search/image.html', form=form, results=result_models)
    else:
        return invalid_request()
