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
    print "================"
    
def randomize_dictionary( path ):
    with open( path, 'r' ) as data_file:
        complete_data = json.loads( data_file.read() )
    counter = 0
    paths = complete_data.keys()
    print len(paths)
    print len(complete_data)
    for path in paths:
        print path, complete_data[ path ]
        counter += 1
        if counter == 5:
            break
    random.shuffle( paths )
    print "==========="  
    counter = 0
    for path in paths:
        print path, complete_data[ path ]
        counter += 1
        if counter == 5:
            break

randomize_dictionary( "data/complete_paths_pos_dict.json" )

        
'''verify_data( "data/test_data_labels_dict.json", "data/train_data_labels_dict.json" )

verify_data( "data/test_data_labels_dict.json", "data/complete_paths_pos_dict.json" )

verify_data( "data/train_data_labels_dict.json", "data/complete_paths_pos_dict.json" )'''
