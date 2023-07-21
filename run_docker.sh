#!/usr/bin/env bash
code_dir=`pwd`
docker run --rm -it --gpus all \
    -v $code_dir:/Variational_Clustering \
    variational_clustering:latest \
    bash -c "cd /Variational_Clustering/ ; bash"