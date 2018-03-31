"""
Predict a set of next nodes in graphichal data (Galaxy workflows) using the trained model
"""

import numpy as np
import os
import json
from keras.models import model_from_json
import re


class PredictNextNode:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.raw_paths = "data/train_data_sequence.txt"
        self.network_config_json_path = "data/model.json"
        self.trained_model_path = "data/weights-epoch-100.hdf5"
        self.data_dictionary = "data/data_dictionary.txt"
        self.data_rev_dict = "data/data_rev_dict.txt"
        self.train_test_labels = "data/multi_labels.txt"

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
    def predict_node( self, trained_model, path_vec, nodes_rev_dict, max_seq_len, top_n=100 ):
        """
        Predict next nodes for a path using a trained model
        """
        top_prediction_prob = dict()
        dimensions = len( nodes_rev_dict )
        path_vec_reshaped = np.reshape( path_vec, ( 1, max_seq_len ) )
        # predict the next tool using the trained model
        prediction = trained_model.predict( path_vec_reshaped, verbose=0 )
        prediction = np.reshape( prediction, ( dimensions, ) )
        # take prediction in reverse order, best ones first
        prediction_pos = np.argsort( prediction, axis=0 )
        top_prediction_pos = prediction_pos[ -top_n: ]
        for index, item in enumerate( reversed( top_prediction_pos ) ):
            top_prediction_prob[ index ] = str( prediction[ item ] )
        # get top n predictions
        top_prediction_pos = prediction_pos[ -top_n: ]
        for index, item in enumerate( reversed( top_prediction_pos ) ):
            top_prediction_prob[ index ] = str( prediction[ item ] )
        # get tool names for the predicted positions
        predicted_nodes = [ nodes_rev_dict[ str( item + 1 ) ] for item in reversed( top_prediction_pos ) ]
        predicted_nodes = ",".join( predicted_nodes )
        path_vec_pos = np.where( path_vec > 0 )[ 0 ]
        path_vec_pos_list = [ str( int( path_vec[ item ] + 1 ) ) for item in path_vec_pos ]
        path_vec_pos_list = ",".join( path_vec_pos_list )
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
        max_seq_len = 40
        all_paths_train = list()
        all_input_seq_paths = dict()
        actual_predicted_nodes = dict()
	actual_labels = list()
        with open( self.raw_paths, 'r' ) as load_all_paths:
            all_paths = load_all_paths.read().split( "\n" )
        for index, item in enumerate( all_paths ):
            all_paths_train.append( item )
        for index, item in enumerate( all_paths_train ):
            # find exact string match, not approximate
            occur = re.findall( '\\b' + input_sequence + '\\b', item )
            if( len( occur ) > 0 ):
                all_input_seq_paths[ index ] = item
        # load the trained model
        loaded_model = self.load_saved_model( self.network_config_json_path, self.trained_model_path )
        nodes_dict = self.get_file_dictionary( self.data_dictionary )
        nodes_rev_dict = self.get_file_dictionary( self.data_rev_dict )
        # pad the input sequence to send it for predicting next nodes
        input_seq_padded = np.zeros( [ max_seq_len ] )
        input_seq_split = input_sequence.split( "," )
        start_pos = max_seq_len - len( input_seq_split )
        for index, item in enumerate( input_seq_split ):
            input_seq_padded[ start_pos + index ] = nodes_dict[ item ] - 1
	predicted_nodes, predicted_prob = self.predict_node( loaded_model, input_seq_padded, nodes_rev_dict, max_seq_len )
	print predicted_nodes
	# find which predicted nodes are present as next nodes
	for item in predicted_nodes.split( "," ):
            next_seq = input_sequence + "," + item
	    for path in all_input_seq_paths:
		pth = all_input_seq_paths[ path ]
		if next_seq in pth:
		    actual_predicted_nodes[ item ] = True
            if not item in actual_predicted_nodes:
		actual_predicted_nodes[ item ] = False
	# get actual next nodes for the input sequence
        with open( self.train_test_labels, 'r' ) as multi_labels:
            multi_train_test_labels = json.loads( multi_labels.read() )
        input_seq_indices = [ str( nodes_dict[ item ] ) for item in input_seq_split ]
        input_seq_indices = ",".join( input_seq_indices )
        print input_seq_indices
	for train_seq, train_label in multi_train_test_labels.iteritems():
	    if train_seq == input_seq_indices:
	        actual_labels = [ nodes_rev_dict[ str( item ) ] for item in train_label.split( "," ) ]
        return { "predicted_nodes": predicted_nodes, "all_input_paths": all_input_seq_paths, "predicted_prob": predicted_prob, "actual_predicted_nodes": actual_predicted_nodes, "actual_labels": actual_labels }
