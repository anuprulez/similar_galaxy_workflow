import json


def check_paths_data( test_paths, train_paths ):
    with open( test_paths, 'r' ) as test_paths_labels:
        test_paths_labels = json.loads( test_paths_labels.read() )
    with open( train_paths, 'r' ) as train_paths_labels:
        train_paths_labels = json.loads( train_paths_labels.read() )
    with open( "data/data_rev_dict.txt", 'r' ) as rev_data_dict:
            reverse_data_dictionary = json.loads( rev_data_dict.read() )
    counter = 0
    counter_test = 0
    for path in test_paths_labels:
        if path in train_paths_labels:
           print path
           path_tools = ",".join( [ reverse_data_dictionary[ str( pos ) ] for pos in path.split( "," ) ] )
           print path_tools
           print test_paths_labels[ path ]
           print train_paths_labels[ path ]
           counter += 1
           print "====="
    print float( counter ) / len( test_paths_labels )

check_paths_data( "data/test_data_labels_dict.json", "data/train_data_labels_dict.json" )

