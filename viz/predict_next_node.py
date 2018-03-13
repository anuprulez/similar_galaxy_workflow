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
        self.raw_file = "data/workflow_steps.txt"
        self.raw_paths = "data/complete_data_sequence.txt"
        self.network_config_json_path = "data/model.json"
        self.trained_model_path = "data/trained_model.hdf5"
        self.graph_vectors_path = "data/doc2vec_model.hdf5"
        self.data_dictionary = "data/data_dictionary.txt"
        self.data_rev_dict = "data/data_rev_dict.txt"
        self.vec_dimension = 100

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
        top_prediction_prob = dict()
        path_vec_reshaped = np.reshape( path_vec, ( 1, 1, len( path_vec ) ) )
        # predict the next tool using the trained model
        prediction = trained_model.predict( path_vec_reshaped, verbose=0 )
        prediction = np.reshape( prediction, ( len( nodes_dict ) ) )
        # take prediction in reverse order, best ones first
        prediction_pos = np.argsort( prediction, axis=0 )
        # get top n predictions
        top_prediction_pos = prediction_pos[ -top_n: ]
        for index, item in enumerate( reversed( top_prediction_pos ) ):
            top_prediction_prob[ index ] = str( prediction[ item ] )
        # get tool names for the predicted positions
        predicted_nodes = [ nodes_rev_dict[ str( item ) ] for item in reversed( top_prediction_pos ) ]
        predicted_nodes = ",".join( predicted_nodes )
        return predicted_nodes, top_prediction_prob

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
        # load the trained model
        loaded_model = self.load_saved_model( self.network_config_json_path, self.trained_model_path )
        # read paths, nodes dictionary from files
        with open( self.raw_paths, 'r' ) as load_all_paths:
            all_paths = load_all_paths.read().split( "\n" )

        nodes_dict = self.get_file_dictionary( self.data_dictionary )
        nodes_rev_dict = self.get_file_dictionary( self.data_rev_dict )
        
        # collect paths and their corresponding vectors
        all_paths = all_paths[ :len( all_paths ) -1 ]
        graph_vectors = h5.File( self.graph_vectors_path, 'r' )
        graph_vectors = graph_vectors[ "doc2vector" ]
        all_paths_train = list()
        all_input_seq_paths = dict()
        input_seq_vec = np.zeros( [ 1, self.vec_dimension ] )
        for index, item in enumerate( all_paths ):
            item = item.split(",")
            item = item[ :len( item ) -1 ]
            all_paths_train.append( ",".join( item ) )
        
        for index, item in enumerate( all_paths_train ):
            if item == input_sequence:
                input_seq_vec = graph_vectors[ index ]
                break
        for index, item in enumerate( all_paths_train ):
            if input_sequence in item: 
                all_input_seq_paths[ index ] = item
        try:
            predicted_nodes, predicted_prob = self.predict_node( loaded_model, input_seq_vec, nodes_dict, nodes_rev_dict )
        except:
            predicted_nodes = {}
            all_input_seq_paths = {}
            predicted_prob = {}
        return { "predicted_nodes": predicted_nodes, "all_input_paths": all_input_seq_paths, "predicted_prob": predicted_prob }
