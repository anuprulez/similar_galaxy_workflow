#!/bin/sh

python retrain_predict_tool.py data/workflows/workflow_connections.tsv config.xml data/generated_files/trained_model_128bs_retrain.hdf5

