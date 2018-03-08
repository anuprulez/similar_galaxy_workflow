"""
Predict a set of next nodes in graphichal data (Galaxy workflows) using the trained model
"""

import sys
import numpy as np
import time
import os
import h5py as h5
import random
import json
from keras.models import model_from_json


class PredictNextNode:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.raw_file = "../data/workflow_steps.txt"
        self.raw_paths = "../data/complete_data_sequence.txt"
        self.network_config_json_path = "../data/model.json"
        self.trained_model_path = "../data/trained_model.hdf5"
        self.graph_vectors_path = "../data/doc2vec_model.hdf5"
        self.data_dictionary = "../data/data_dictionary.txt"
        self.data_rev_dict = "../data/data_rev_dict.txt"

    @classmethod
    def load_saved_model( self, network_config_path, weights_path ):
        """
        Load the saved trained model using the saved network and its weights
        """
        with open( network_config_path, 'r' ) as network_config_file:
            loaded_model = network_config_file.read()
        # load the network
        loaded_model = model_from_json(loaded_model)
        # load the saved weights into the model
        loaded_model.load_weights( weights_path )
        return loaded_model

    @classmethod
    def predict_node( self, trained_model, path_vec, nodes_dict, nodes_rev_dict, top_n=5 ):
        """
        Predict next nodes for a path using a trained model 
        """
        path_vec_reshaped = np.reshape( path_vec, ( 1, 1, len( path_vec ) ) )
        # predict the next tool using the trained model
        prediction = trained_model.predict( path_vec_reshaped, verbose=0 )
        prediction = np.reshape( prediction, ( len( nodes_dict ) ) )
        # take prediction in reverse order, best ones first
        prediction_pos = np.argsort( prediction, axis=0 )
        # get top n predictions
        top_prediction_pos = prediction_pos[ -top_n: ]
        # get tool names for the predicted positions
        predicted_nodes = [ nodes_rev_dict[ str( item ) ] for item in top_prediction_pos ]
        print ("Predicted nodes for the input sequence...")
        print predicted_nodes

    @classmethod
    def find_next_nodes( self ):
        """
        Find a set of possible next nodes
        """
        # load the trained model
        loaded_model = self.load_saved_model( self.network_config_json_path, self.trained_model_path )
        # read paths, nodes dictionary from files
        with open( self.raw_paths, 'r' ) as load_all_paths:
            all_paths = load_all_paths.read().split( "\n" )
        with open( self.data_dictionary, 'r' ) as data_dict:
            nodes_dict = json.loads( data_dict.read() )
        with open( self.data_rev_dict, 'r' ) as data_rev_dict:
            nodes_rev_dict = json.loads( data_rev_dict.read() )
        # collect paths and their corresponding vectors
        all_paths = all_paths[ :len( all_paths ) -1 ]
        graph_vectors = h5.File( self.graph_vectors_path, 'r' )
        graph_vectors = graph_vectors[ "doc2vector" ]
        random_sample_pos = random.randint( 0, len( all_paths ) )
        input_seq = all_paths[ random_sample_pos ]
        input_seq_vec = graph_vectors[ random_sample_pos ]
        # print all the paths containing this input sequence
        print("All the paths containing the input sequence: %s", input_seq)
        for item in all_paths:
            if input_seq in item:
                print item
        # predict next node for random vectors
        self.predict_node( loaded_model, input_seq_vec, nodes_dict, nodes_rev_dict )
            

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_next_node.py" )
        exit( 1 )
    start_time = time.time()
    evaluate_perf = PredictNextNode()
    evaluate_perf.find_next_nodes()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ) )
