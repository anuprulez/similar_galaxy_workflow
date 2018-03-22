"""
Predict a set of next nodes in graphichal data (Galaxy workflows) using the trained model
"""

import numpy as np
import os
import json
from keras.models import model_from_json
import yaml
import requests

NAME     = "name"
CATEGORY = "tool_panel_section_label"
TOOL_REPO_URL = "https://raw.githubusercontent.com/usegalaxy-eu/usegalaxy-eu-tools/master/tools.yaml"
TOOL_LIST     = "tools"


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
    def predict_node( self, trained_model, path_vec, nodes_rev_dict, max_seq_len, top_n=10 ):
        """
        Predict next nodes for a path using a trained model
        """
        top_prediction_prob = dict()
        dimensions = len( nodes_rev_dict )
        path_vec_reshaped = np.reshape( path_vec, ( 1, max_seq_len ) )
        with open( self.train_test_labels, 'r' ) as train_data_labels:
            data_labels = json.loads( train_data_labels.read() )
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
        top_pred_rev = [ str( item + 1 ) for item in reversed( top_prediction_pos ) ]
        #print( "Predicted labels: %s" % ( ",".join( top_pred_rev ) ) )
        path_vec_pos = np.where( path_vec > 0 )[ 0 ]
        path_vec_pos_list = [ str( int( path_vec[ item ] + 1 ) ) for item in path_vec_pos ]
        path_vec_pos_list = ",".join( path_vec_pos_list )
        #if path_vec_pos_list in data_labels:
            #print ( "Actual labels: %s" % ( data_labels[ path_vec_pos_list ] ) )
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
        max_seq_len = 125 # max length of training input
        all_paths_train = list()
        all_input_seq_paths = dict()
        with open( self.raw_paths, 'r' ) as load_all_paths:
            all_paths = load_all_paths.read().split( "\n" )
        all_paths = all_paths[ :len( all_paths ) -1 ]
        for index, item in enumerate( all_paths ):
            item = item.split( "," )
            item = item[ :len( item ) - 1 ]
            all_paths_train.append( ",".join( item ) )
        for index, item in enumerate( all_paths_train ):
            if input_sequence in item: 
                all_input_seq_paths[ index ] = item

        # load the trained model
        loaded_model = self.load_saved_model( self.network_config_json_path, self.trained_model_path )
        nodes_dict = self.get_file_dictionary( self.data_dictionary )
        nodes_rev_dict = self.get_file_dictionary( self.data_rev_dict )

        input_seq_padded = np.zeros( [ max_seq_len ] )
        input_seq_split = input_sequence.split( "," )
        start_pos = max_seq_len - len( input_seq_split )
        for index, item in enumerate( input_seq_split ):
            input_seq_padded[ start_pos + index ] = nodes_dict[ item ] - 1
        try:
            predicted_nodes, predicted_prob = self.predict_node( loaded_model, input_seq_padded, nodes_rev_dict, max_seq_len )
        except Exception as exception:
            print( exception )
            predicted_nodes = {}
            all_input_seq_paths = {}
            predicted_prob = {}
        return { "predicted_nodes": predicted_nodes, "all_input_paths": all_input_seq_paths, "predicted_prob": predicted_prob }

    @classmethod
    def get_tool_category_catalog( self ):
        """
        Retrieves all tools' categories from usegalaxy.eu
        """
        tool_categories = {}

        response = requests.get(TOOL_REPO_URL)

        if response.status_code == 200:
            tool_repo = yaml.load(response.content)

            # we extract the tool list from the retrieved tool repository
            tool_list = tool_repo[TOOL_LIST]

            # we collect each tool and corresponding category
            for tool in tool_list:
                tool_categories[tool[NAME]] = tool[CATEGORY]

        return tool_categories
