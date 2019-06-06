#!/bin/sh

python predict_tool.py -wf data/workflows/wf-connections.tsv -cf config.xml -tm data/generated_files/trained_model.hdf5 -tu data/tool_usage/tool-popularity.tsv -cd '2017-12-01' -pl 25 -ld 'data/generated_files'
