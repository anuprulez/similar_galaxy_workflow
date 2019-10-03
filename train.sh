#!/bin/bash

python scripts/main.py -wf data/wf-connections-19-03.tsv -tu data/tool-popularity-19-03.tsv -om data/trained_model_sample.hdf5 -cd '2017-12-01' -pl 25 -ep 10 -oe 10 -me 30 -ts 0.0 -vs 0.2 -bs '1,128' -ut '1,128' -es '1,128' -dt '0.0,0.5' -sd '0.0,0.5' -rd '0.0,0.5' -lr '0.00001, 0.1' -ar 'elu' -ao 'sigmoid'
