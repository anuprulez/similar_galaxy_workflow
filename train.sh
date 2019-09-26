#!/bin/bash

python scripts/main.py -wf data/wf_connections_0_2k.tsv -cf config.xml -tm data/trained_model_sample.hdf5 -tu data/tool-popularity.tsv -cd '2017-12-01' -pl 25
