#!/bin/sh

python predict_tool.py data/workflows/wf_connections_0_6k.tsv config.xml data/generated_files/trained_model.hdf5 data/tool_usage/tool-popularity.tsv '2017-12-01'
