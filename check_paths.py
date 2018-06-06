import json
import csv


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

def merge_tabular_files( files_list ):
    merged_file = list()
    new_file = "data/merged_data_file.tsv"
    fields = [ 'wf_id', "in_id", "in_tool", "in_tool_v", "out_id", "out_tool", "out_tool_v" ]
    merged_file.append( fields )
    for file_index, file_path in enumerate( files_list ):
        print( "Processing file: %d:%s" % ( file_index, file_path ) )
        with open( file_path, 'rt' ) as workflow_connections_file:
            workflow_connections = csv.reader( workflow_connections_file, delimiter=',' )
            for index, row in enumerate( workflow_connections ):
                if not index:
                    continue
                row_split = row[0].split( "\t" )
                row_split[ 0 ] = "sheet_" + str( file_index + 1 ) + "_" + str( row_split[ 0 ] )
                merged_file.append( row_split )
    with open(new_file, "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows(merged_file)
  
merge_tabular_files( [ "data/workflow_connections.tsv", "data/workflow_connections_2.tsv" ] )
#check_paths_data( "data/test_data_labels_dict.json", "data/train_data_labels_dict.json" )




