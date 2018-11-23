import json
import random


def verify_data( test_path, train_path ):
    with open( test_path, 'r' ) as test_labels:
        test_data = json.loads( test_labels.read() )
    with open( train_path, 'r' ) as train_labels:
        train_data = json.loads( train_labels.read() )
    counter = 0
    print len(test_data)
    print len(train_data)
    for path in test_data:
        if path in train_data:
            counter += 1
    print( "Percentage overlap in train and test data: %.2f" % ( counter ) )

verify_data( "data/test_data_labels_dict.json", "data/train_data_labels_dict.json" )
verify_data( "data/test_data_labels_names_dict.json", "data/train_data_labels_names_dict.json" )
