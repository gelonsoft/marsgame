#!/bin/bash

# Start TensorBoard and serve it on port 6006
docker run -it --rm -p 6006:6006 -v $(pwd)/runs:/data/runs tensorflow/tensorflow tensorboard --logdir=/data/runs --port=6006