"""
Predict a set of next nodes in graphichal data (Galaxy workflows) using the trained model
"""

import numpy as np
import os
import json
from keras.models import model_from_json
import yaml
import requests
import operator

NAME     = "name"
CATEGORY = "tool_panel_section_label"
TOOL_REPO_URL = "https://raw.githubusercontent.com/usegalaxy-eu/usegalaxy-eu-tools/master/tools.yaml"
TOOL_LIST     = "tools"
TOP_N = 10

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
    def predict_node( self, trained_model, path_vec, similar_tools, nodes_rev_dict, nodes_dict, max_seq_len ):
        """
        Predict next nodes for a path using a trained model
        """
        predicted_nodes_list = list()
        padded_sequences = list()
        ordered_nodes = dict()
        padded_sequences.append( path_vec )
        for item in similar_tools:
            if item in nodes_dict:
                padded_seq = self.get_padded_input( max_seq_len, item, nodes_dict )
                padded_sequences.append( padded_seq )
        for item in padded_sequences:
            predicted_nodes = self.predict_by_model( trained_model, item, nodes_rev_dict, max_seq_len )
            predicted_nodes_list.extend( predicted_nodes )
        for node in predicted_nodes_list:
            if node in ordered_nodes:
                ordered_nodes[ node ] += 1
            else:
                ordered_nodes[ node ] = 1
        ordered_nodes = sorted( ordered_nodes.items(), key=operator.itemgetter( 1 ), reverse=True )[ :TOP_N ]
        ordered_nodes = [ item for ( item, freq ) in ordered_nodes ]
        ordered_nodes = ",".join( ordered_nodes )
        return ordered_nodes

    @classmethod
    def predict_by_model( self, trained_model, path_vec, nodes_rev_dict, max_seq_len, top_n=TOP_N ):
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
        return predicted_nodes

    @classmethod
    def get_file_dictionary( self, file_name ):
        """
        Get a dictionary for tools
        """
        with open( file_name, 'r' ) as data_dict:
            nodes_dict = json.loads( data_dict.read() )
        return nodes_dict

    @classmethod
    def get_padded_input( self, max_seq_len, input_seq, nodes_dict ):
        """
        Make 0 padded vector for an input sequence
        """
        input_seq_padded = np.zeros( [ max_seq_len ] )
        input_seq_split = input_seq.split( "," )
        start_pos = max_seq_len - len( input_seq_split )
        for index, item in enumerate( input_seq_split ):
            input_seq_padded[ start_pos + index ] = nodes_dict[ item ] - 1
        return input_seq_padded

    @classmethod
    def get_similar_tools( self, input_sequence ):
        """
        Get similar tools using Galaxy's manually labeled categories
        """
        similar_tools = list()
        last_tool_category = ""
        tool_categories = self.get_tool_category_catalog()
        last_tool = input_sequence.split( "," )[ -1 ]
        if last_tool in tool_categories:
            last_tool_category = tool_categories[ last_tool ]
        for item in tool_categories:
            if item in tool_categories and last_tool_category == tool_categories[ item ]:
                similar_tools.append( item )
        return similar_tools

    @classmethod
    def find_next_nodes( self, input_sequence="" ):
        """
        Find a set of possible next nodes
        """
        max_seq_len = 125 # max length of training input
        all_paths_train = list()
        all_input_seq_paths = dict()
        # load the trained model
        loaded_model = self.load_saved_model( self.network_config_json_path, self.trained_model_path )
        nodes_dict = self.get_file_dictionary( self.data_dictionary )
        nodes_rev_dict = self.get_file_dictionary( self.data_rev_dict )
        input_seq_padded = self.get_padded_input( max_seq_len, input_sequence, nodes_dict )
        similar_tools = self.get_similar_tools( input_sequence )
        predicted_nodes = self.predict_node( loaded_model, input_seq_padded, similar_tools, nodes_rev_dict, nodes_dict, max_seq_len )
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
        return { "predicted_nodes": predicted_nodes, "all_input_paths": all_input_seq_paths }

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
