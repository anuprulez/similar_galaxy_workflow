"""
Predict nodes in graphichal data (Galaxy workflows) using Recurrent Neural Network (LSTM)
"""

import os
import collections
import numpy as np
import json


class PrepareData:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.raw_file = self.current_working_dir + "/data/workflow_steps.txt"
        self.train_file = self.current_working_dir + "/data/train_data.txt"
        self.sequence_file = self.current_working_dir + "/data/train_data_sequence.txt"
        self.data_dictionary = self.current_working_dir + "/data/data_dictionary.txt"
        self.data_rev_dict = self.current_working_dir + "/data/data_rev_dict.txt"
        self.multi_train_labels = self.current_working_dir + "/data/multi_labels.txt"
        self.max_tool_sequence_len = 40

    @classmethod
    def process_processed_data( self, fname ):
        """
        Get all the tools and complete set of individual paths for each workflow
        """
        tokens = list()
        raw_paths = list()
        with open( fname ) as f:
            data = f.readlines()
        raw_paths = [ x.replace( "\n", '' ) for x in data ]
        for item in raw_paths:
            split_items = item.split( " " )
            for token in split_items:
                if token not in tokens:
                    tokens.append( token )
        tokens = np.array( tokens )
        tokens = np.reshape( tokens, [ -1, ] )
        return tokens, raw_paths

    @classmethod
    def create_data_dictionary( self, words ):
        """
        Create two dictionaries having tools names and their indexes
        """
        count = collections.Counter( words ).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[ word ] = len( dictionary ) + 1
        reverse_dictionary = dict( zip( dictionary.values(), dictionary.keys() ) )
        with open( self.data_dictionary, 'w' ) as data_dict:
            data_dict.write( json.dumps( dictionary ) )
        with open( self.data_rev_dict, 'w' ) as data_rev_dict:
            data_rev_dict.write( json.dumps( reverse_dictionary ) )
        return dictionary, reverse_dictionary

    @classmethod
    def create_train_labels_file( self, dictionary, raw_paths ):
        """
        Create training data with its labels with varying window sizes
        """
        train_data = list()
        train_data_sequence = list()
        for index, item in enumerate( raw_paths ):
            tools = item.split(" ")
            len_tools = len( tools )
            if len_tools <= self.max_tool_sequence_len:
                for window in range( 1, len_tools ):
                    training_sequence = tools[ 0: window + 1 ]
                    tools_pos = [ str( dictionary[ str( tool_item ) ] ) for tool_item in training_sequence ]
                    if len( tools_pos ) > 1:
                        tools_pos = ",".join( tools_pos )
                        data_seq = ",".join( training_sequence )
                        if tools_pos not in train_data:
                            train_data.append( tools_pos )
                        if data_seq not in train_data_sequence:
                            train_data_sequence.append( data_seq )
                print ( "Path %d processed" % ( index + 1 ) )
            else:
                print ( "Path %d excluded due to exceeded length" % ( index + 1 ) )
        with open( self.train_file, "w" ) as train_file:
            for item in train_data:
                train_file.write( "%s\n" % item )
        with open( self.sequence_file, "w" ) as train_seq:
            for item in train_data_sequence:
                train_seq.write( "%s\n" % item )

    @classmethod
    def prepare_train_test_data( self ):
        """
        Read training data and its labels files
        """
        train_file = open( self.train_file, "r" )
        train_file = train_file.read().split( "\n" )
        len_data = len( train_file )
        train_multi_label_samples = dict()
        for item in train_file:
            if item and item not in "":
                tools = item.split( "," )
                label = tools[ -1 ]
                train_tools = tools[ :len( tools) - 1 ]
                train_tools = ",".join( train_tools )
                len_train_seq = len( train_tools.split( "," ) )
                if train_tools in train_multi_label_samples:
                    train_multi_label_samples[ train_tools ] += "," + tools[ -1 ]
                else:
                    train_multi_label_samples[ train_tools ] = tools[ -1 ]
        with open( self.multi_train_labels, 'w' ) as train_multilabel_file:
            train_multilabel_file.write( json.dumps( train_multi_label_samples ) )
        return train_multi_label_samples

    @classmethod
    def read_data( self ):
        """
        Convert the data into corresponding arrays
        """
        processed_data, raw_paths = self.process_processed_data( self.raw_file )
        dictionary, reverse_dictionary = self.create_data_dictionary( processed_data )
        self.create_train_labels_file( dictionary, raw_paths )
        # all the nodes/tools are classes as well 
        train_labels_data = self.prepare_train_test_data()
        num_classes = len( dictionary )
        len_train_data = len( train_labels_data )
        # initialize the training data matrix
        train_data_array = np.zeros( [ len_train_data, self.max_tool_sequence_len ] )
        train_label_array = np.zeros( [ len_train_data, num_classes ] )
        train_counter = 0
        for train_seq, train_label in list( train_labels_data.items() ):
            positions = train_seq.split( "," )
            start_pos = self.max_tool_sequence_len - len( positions )
            for id_pos, pos in enumerate( positions ):
                train_data_array[ train_counter ][ start_pos + id_pos ] = int( pos ) - 1
            for label_item in train_label.split( "," ):
                train_label_array[ train_counter ][ int( label_item ) - 1 ] = 1.0
            train_counter += 1
        return train_data_array, train_label_array, dictionary, reverse_dictionary
