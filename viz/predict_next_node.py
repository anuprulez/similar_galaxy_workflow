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
        self.raw_paths = "data/complete_data_sequence.txt"
        self.network_config_json_path = "data/model.json"
        self.trained_model_path = "data/trained_model.hdf5"
        self.data_dictionary = "data/data_dictionary.txt"
        self.data_rev_dict = "data/data_rev_dict.txt"

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
        dimensions = len( path_vec )
        path_vec_reshaped = np.reshape( path_vec, ( 1, dimensions ) )
        # predict the next tool using the trained model
        prediction = trained_model.predict( path_vec_reshaped, verbose=0 )
        prediction = np.reshape( prediction, ( dimensions, ) )
        # take prediction in reverse order, best ones first
        prediction_pos = np.argsort( prediction, axis=0 )
        # get top n predictions
        top_prediction_pos = prediction_pos[ :top_n ]
        # get tool names for the predicted positions
        print top_prediction_pos
        predicted_nodes = [ nodes_rev_dict[ str( item ) ] for item in top_prediction_pos ]
        return ",".join( predicted_nodes )

    @classmethod
    def get_file_dictionary( self, file_name ):
        """
        Get a dictionary for tools
        """
        with open( file_name, 'r' ) as data_dict:
            nodes_dict = json.loads( data_dict.read() )
        return nodes_dict

    @classmethod
    def find_next_nodes( self, input_sequence="" ):
        """
        Find a set of possible next nodes
        """
        input_seq_paths = dict()
        all_paths_train = list()
        all_input_seq_paths = dict()
        with open( self.raw_paths, 'r' ) as load_all_paths:
            all_paths = load_all_paths.read().split( "\n" )
        all_paths = all_paths[ :len( all_paths ) -1 ]
        for index, item in enumerate( all_paths ):
            item = item.split(",")
            item = item[ :len( item ) -1 ]
            all_paths_train.append( ",".join( item ) )
        for index, item in enumerate( all_paths_train ):
            if input_sequence in item: 
                all_input_seq_paths[ index ] = item

        # load the trained model
        loaded_model = self.load_saved_model( self.network_config_json_path, self.trained_model_path )
        nodes_dict = self.get_file_dictionary( self.data_dictionary )
        nodes_rev_dict = self.get_file_dictionary( self.data_rev_dict )
        input_seq_padded = np.zeros( [ len( nodes_dict ) ] )
        input_seq_split = input_sequence.split( "," )
        print input_sequence
        start_pos = len( nodes_dict ) - len( input_seq_split )
        for index, item in enumerate( input_seq_split ):
            input_seq_padded[ start_pos + index ] = nodes_dict[ item ]
        print input_seq_padded
        try:
            predicted_nodes = self.predict_node( loaded_model, input_seq_padded, nodes_dict, nodes_rev_dict )
        except Exception as exception:
            print exception
            predicted_nodes = {}
            all_input_seq_paths = {}
        return { "predicted_nodes": predicted_nodes, "all_input_paths": all_input_seq_paths }
