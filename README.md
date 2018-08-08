# Flask Semantic Image Search
A simple and fast semantic image search engine using convolutional neural networks.
Quickly spin up a microservice with your own model and/or images using our Docker images.

## Requirements
- [Docker](https://docs.docker.com/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)
  - `pip install docker-compose`

Allocate Docker enough RAM or you may get unhelpful error messages.
A good rule of thumb is 2GB + the disk size of `search_index.pkl` ([see below](#indexing-images)).

## Prerequisites
Two things are needed before running this service, a trained Keras model and indexed images.
Use the following companion repo and follow the steps below.<br />
[keras_image_similarity_training](https://github.com/wayfair/keras_image_similarity_training)

### Import Keras Model
Tensorflow Serving needs files from Keras to load the image similarity model.
Obtain a trained Keras model and export it to the SavedModel format.
A public trained model is available for usage from Keras's model zoo. You may also
train a model on domain specific labeled images for the better results.

After exporting a model, the directory `savedmodels/inception_resnet_v2/` should appear in `keras_image_similarity_training`.
Assuming both repos are in the same directory, copy the exported models to this repo like so
```bash
cp -r keras_image_similarity_training/savedmodels/inception_resnet_v2/ \
      flask_semantic_image_search/tensorflow_serving/models
```

### Indexing Images
All search-able images need to be embedded with the trained Keras model. The image
embeddings are used to build a fast kNN search structure. Using the `keras_image_similarity_training` repo,
follow the instructions there to create one and pickle it.

After indexing all images, the file `search_index.pkl` should appear in `keras_image_similarity_training`.
Assuming both repos are in the same directory, copy the pickle over to this repo like so
```bash
cp keras_image_similarity_training/search_index.pkl \
   flask_semantic_image_search/data/
```

### Provide Image Files (Optional)
To use the minimal front-end provided in this microservice, the Flask app will need access to the
raw image files. Put the images in `flask_semantic_image_search/data/images`. The directory can be structured
however you want, just be consistence with the relative paths in the `labeled_images.json` used in
`keras_image_similarity_training`.

## Usage
Runs the service as Gunicorn Flask app with multiple workers. Creates a docker image from the current
source code, and does not allow for live code changes. To reflect code changes, rebuild the docker images.
```bash
docker-compose up
```
Runs the service with the Flask development server. Creates a docker image and mounts the
current source code, which allows for live code changes.
```bash
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```
Test to see if the service is running [here](http://localhost:9090/search/).

## Cleanup
To stop the service for now, run the following
```bash
docker-compose down
```
To stop the service and remove the containers and images, run the following
```bash
docker-compose down --rmi all
```

## Endpoints
- Image Search
  - [http://localhost:9090/search/image/](http://localhost:9090/search/image/)
  - `[POST]`
    - `image (file)`: Image file
    - `rows (int)`: Number of results to return
    - `format (str)`: JSON or HTML

## Caveats
- Scikit-Learn's `BallTree` implementation stores raw data as `float64`. Convolution neural networks commonly use `float32`,
so `BallTrees` use twice as much memory than needed. We recommend using another kNN library or compiling Scikit-Learn's `neighbors`
module with `float32` data structures.
- By default Gunicorn spawns workers by forking a master process then creating a `Flask` app within each worker.
This means the search index would be repeatedly loaded into each worker, which quickly depletes memory.
Gunicorn's `--preload` command arg gets around this by creating the `Flask` app in the master process once,
then forking it to spawn each worker.
- Tensorflow Serving can be compiled with CUDA and/or CPU instruction sets for more efficient processing. For convenience,
we are using a generic build of Tensorflow Serving for CPU only. For faster response times, build it with CUDA and/or CPU instruction sets.
