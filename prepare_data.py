"""
Predict nodes in graphichal data (Galaxy workflows) using Recurrent Neural Network (LSTM)
"""

import os
import collections
import numpy as np
import json
import random


class PrepareData:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.raw_file = self.current_working_dir + "/data/workflow_steps.txt"
        self.data_dictionary = self.current_working_dir + "/data/data_dictionary.txt"
        self.data_rev_dict = self.current_working_dir + "/data/data_rev_dict.txt"
        self.train_file = self.current_working_dir + "/data/train_data.txt"
        self.train_sequence_file = self.current_working_dir + "/data/train_data_sequence.txt"
        self.test_file = self.current_working_dir + "/data/test_data.txt"
        self.test_sequence_file = self.current_working_dir + "/data/test_data_sequence.txt"
        self.train_data_labels_dict = self.current_working_dir + "/data/train_data_labels_dict.txt"
        self.test_data_labels_dict = self.current_working_dir + "/data/test_data_labels_dict.txt"
        self.tool_filetypes = self.current_working_dir + "/data/tools_file_types.json"
        self.compatible_tools_filetypes = self.current_working_dir + "/data/compatible_tools.json"
        self.max_tool_sequence_len = 40
        self.test_share = 0.4

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
            split_items = item.split( "," )
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
    def process_train_paths( self, train_paths, dictionary ):
        """
        Process train paths using a variable length sliding window
        """
        train_data = list()
        train_data_sequence = list()
        random.shuffle( train_paths )
        for index, item in enumerate( train_paths ):
            tools = item.split( "," )
            len_tools = len( tools )
            if len_tools <= self.max_tool_sequence_len:
                for pos in range( len_tools ):
                    for window in range( 1, len_tools ):
                        sequence = tools[ pos: window + pos + 1 ]
                        tools_pos = [ str( dictionary[ str( tool_item ) ] ) for tool_item in sequence ]
                        if len( tools_pos ) > 1:
                            tools_pos = ",".join( tools_pos )
                            data_seq = ",".join( sequence )
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
        with open( self.train_sequence_file, "w" ) as train_seq:
            for item in train_data_sequence:
                train_seq.write( "%s\n" % item )

    @classmethod
    def process_test_paths( self, test_paths, dictionary ):
        """
        Process test paths of variable length keeping the first tool/node fixed
        """
        test_data = list()
        test_data_sequence = list()
        random.shuffle( test_paths )
        for index, item in enumerate( test_paths ):
            tools = item.split( "," )
            len_tools = len( tools )
            if len_tools <= self.max_tool_sequence_len:
                for window in range( 1, len_tools ):
                    sequence = tools[ 0: window + 1 ]
                    tools_pos = [ str( dictionary[ str( tool_item ) ] ) for tool_item in sequence ]
                    if len( tools_pos ) > 1:
                        tools_pos = ",".join( tools_pos )
                        data_seq = ",".join( sequence )
                        if tools_pos not in test_data:
                            test_data.append( tools_pos )
                        if data_seq not in test_data_sequence:
                            test_data_sequence.append( data_seq )
                print ( "Path %d processed" % ( index + 1 ) )
            else:
                print ( "Path %d excluded due to exceeded length" % ( index + 1 ) )
        with open( self.test_file, "w" ) as test_file:
            for item in test_data:
                test_file.write( "%s\n" % item )
        with open( self.test_sequence_file, "w" ) as test_seq:
            for item in test_data_sequence:
                test_seq.write( "%s\n" % item )

    @classmethod
    def prepare_paths_labels_dictionary( self, read_file, destination_file, filetype_compatibility, dictionary, reverse_dictionary ):
        """
        Create a dictionary of sequences with their labels for training and test paths
        """
        paths = open( read_file, "r" )
        paths = paths.read().split( "\n" )
        paths_labels = dict()
        random.shuffle( paths )
        for item in paths:
            if item and item not in "":
                all_labels = list()
                tools = item.split( "," )
                label = tools[ -1 ]
                train_tools = tools[ :len( tools ) - 1 ]
                last_tool = reverse_dictionary[ int( train_tools[ -1 ] ) ]
                train_tools = ",".join( train_tools )
                all_labels.append( label )
                # encode the compatible next tools along with the labels
                '''if last_tool in dictionary and last_tool in filetype_compatibility:
                    last_tool_name = dictionary[ last_tool ]
                    compatible_next_tools = filetype_compatibility[ last_tool ].split( "," )
                    for tool in compatible_next_tools:
                        if tool in dictionary:
                            all_labels.append( str( dictionary[ tool ] ) )'''
                paths_labels[ train_tools ] = ",".join( list( set( all_labels ) ) )
        with open( destination_file, 'w' ) as multilabel_file:
            multilabel_file.write( json.dumps( paths_labels ) )
        return paths_labels

    @classmethod
    def pad_paths( self, paths_dictionary, num_classes ):
        """
        Add padding to the tools sequences and create multi-hot encoded labels
        """
        size_data = len( paths_dictionary )
        data_mat = np.zeros( [ size_data, self.max_tool_sequence_len ] )
        label_mat = np.zeros( [ size_data, num_classes ] )
        train_counter = 0
        for train_seq, train_label in list( paths_dictionary.items() ):
            positions = train_seq.split( "," )
            start_pos = self.max_tool_sequence_len - len( positions )
            for id_pos, pos in enumerate( positions ):
                data_mat[ train_counter ][ start_pos + id_pos ] = int( pos ) - 1
            for label_item in train_label.split( "," ):
                label_mat[ train_counter ][ int( label_item ) - 1 ] = 1.0
            train_counter += 1
        return data_mat, label_mat

    @classmethod
    def assign_filetype_compatibility( self, filetypes_path, dictionary ):
        """
        Get the tools with compatible file types
        """
        max_compatible = 100
        with open( filetypes_path, "r" ) as file_types:
            tools_filetypes = json.loads( file_types.read() )
        tools_compatibility = dict()
        for out_tool in tools_filetypes:
            output_types = tools_filetypes[ out_tool ][ "output_types" ]
            compatible_next_tools = list()
            if len( output_types ) > 0:
                for in_tool in tools_filetypes:
                    if out_tool != in_tool:
                        input_types = tools_filetypes[ in_tool ][ "input_types" ]
                        if len( input_types ) > 0:
                            compatible_filetypes = [ filetype for filetype in output_types if filetype in input_types ]
                            if len( compatible_filetypes ) > 0:
                                if out_tool in dictionary and in_tool in dictionary:
                                    if in_tool not in compatible_next_tools:
                                        compatible_next_tools.append( in_tool )
            tools_compatibility[ out_tool ] = ",".join( compatible_next_tools[ :max_compatible ] )
        with open( self.compatible_tools_filetypes, "w" ) as compatible_filetypes:
            compatible_filetypes.write( json.dumps( tools_compatibility ) )
        return tools_compatibility

    @classmethod
    def get_data_labels_mat( self ):
        """
        Convert the training and test paths into corresponding numpy matrices
        """
        processed_data, raw_paths = self.process_processed_data( self.raw_file )
        dictionary, reverse_dictionary = self.create_data_dictionary( processed_data )
        num_classes = len( dictionary )
        # randomize all the paths
        random.shuffle( raw_paths )
        next_compatible_tools = self.assign_filetype_compatibility( self.tool_filetypes, dictionary )
        # divide train and test paths
        test_share = self.test_share * len( raw_paths )
        test_paths = raw_paths[ :int( test_share ) ]
        train_paths = raw_paths[ int( test_share ): ]
        random.shuffle( train_paths )
        print( "Processing train paths..." )
        self.process_train_paths( train_paths, dictionary )
        print( "Processing test paths..." )
        self.process_test_paths( test_paths, dictionary )
        # create sequences with labels for train and test paths
        train_paths_dict = self.prepare_paths_labels_dictionary( self.train_file, self.train_data_labels_dict, next_compatible_tools, dictionary, reverse_dictionary )
        test_paths_dict = self.prepare_paths_labels_dictionary( self.test_file, self.test_data_labels_dict, next_compatible_tools, dictionary, reverse_dictionary )
        # create 0 padded sequences from train and test paths
        train_data, train_labels = self.pad_paths( train_paths_dict, num_classes )
        test_data, test_labels = self.pad_paths( test_paths_dict, num_classes )
        return train_data, train_labels, test_data, test_labels, dictionary, reverse_dictionary, next_compatible_tools
