#!/bin/bash

python scripts/main.py -wf data/workflow-connections-19-09.tsv -tu data/tool-popularity-19-09.tsv -om data/tool_recommendation_model.hdf5 -cd '2017-12-01' -pl 25 -ep 10 -oe 10 -me 30 -ts 0.2 -vs 0.2 -bs '1,512' -ds '1,512' -fs '1,512' -es '1,512' -ks '2,10' -dt '0.0,0.5' -sd '0.0,0.5' -lr '0.00001,0.1' -da 'elu' -oa 'sigmoid' -cpus 16
