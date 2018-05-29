"""
Prepare the workflows to be used by downstream machine learning algorithms
"""

import os
import collections
import numpy as np
import json
import random


class PrepareData:

    @classmethod
    def __init__( self, max_seq_length, test_data_share ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.raw_file = self.current_working_dir + "/data/workflow_steps.txt"
        self.data_dictionary = self.current_working_dir + "/data/data_dictionary.txt"
        self.data_rev_dict = self.current_working_dir + "/data/data_rev_dict.txt"
        self.train_file = self.current_working_dir + "/data/train_data.txt"
        self.train_sequence_file = self.current_working_dir + "/data/train_data_sequence.txt"
        self.test_file = self.current_working_dir + "/data/test_data.txt"
        self.test_sequence_file = self.current_working_dir + "/data/test_data_sequence.txt"
        self.test_actual_file = self.current_working_dir + "/data/test_actual_data.txt"
        self.test_actual_sequence_file = self.current_working_dir + "/data/test_actual_data_sequence.txt"
        self.train_data_labels_dict = self.current_working_dir + "/data/train_data_labels_dict.txt"
        self.test_data_labels_dict = self.current_working_dir + "/data/test_data_labels_dict.txt"
        self.test_actual_data_labels_dict = self.current_working_dir + "/data/test_actual_data_labels_dict.txt"
        self.compatible_tools_filetypes = self.current_working_dir + "/data/compatible_tools.json"
        self.max_tool_sequence_len = max_seq_length
        self.test_share = test_data_share

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
    def decompose_paths( self, paths, dictionary, file_pos, file_names ):
        """
        Decompose the paths to variable length sub-paths keeping the first tool fixed
        """
        sub_paths_pos = list()
        sub_paths_names = list()
        for index, item in enumerate( paths ):
            tools = item.split( "," )
            len_tools = len( tools )
            if len_tools <= self.max_tool_sequence_len:
                for window in range( 1, len_tools ):
                    sequence = tools[ 0: window + 1 ]
                    tools_pos = [ str( dictionary[ str( tool_item ) ] ) for tool_item in sequence ]
                    if len( tools_pos ) > 1:
                        tools_pos = ",".join( tools_pos )
                        data_seq = ",".join( sequence )
                        if tools_pos not in sub_paths_pos:
                            sub_paths_pos.append( tools_pos )
                        if data_seq not in sub_paths_names:
                            sub_paths_names.append( data_seq )
        with open( file_pos, "w" ) as sub_paths_file_pos:
            for item in sub_paths_pos:
                sub_paths_file_pos.write( "%s\n" % item )
        with open( file_names, "w" ) as sub_paths_file_names:
            for item in sub_paths_names:
                sub_paths_file_names.write( "%s\n" % item )

    @classmethod
    def take_actual_paths( self, paths, dictionary, file_pos, file_names ):
        """
        Take paths as such. No decomposition.
        """
        sub_paths_pos = list()
        sub_paths_names = list()
        for index, item in enumerate( paths ):
            sequence = item.split( "," )
            if len( sequence ) <= self.max_tool_sequence_len:
                tools_pos = [ str( dictionary[ str( tool_item ) ] ) for tool_item in sequence ]
                if len( tools_pos ) > 1:
                    tools_pos = ",".join( tools_pos )
                    data_seq = ",".join( sequence )
                    sub_paths_pos.append( tools_pos )
                    sub_paths_names.append( data_seq )
        with open( file_pos, "w" ) as sub_paths_file_pos:
            for item in sub_paths_pos:
                sub_paths_file_pos.write( "%s\n" % item )
        with open( file_names, "w" ) as sub_paths_file_names:
            for item in sub_paths_names:
                sub_paths_file_names.write( "%s\n" % item )

    @classmethod
    def prepare_paths_labels_dictionary( self, read_file, destination_file ):
        """
        Create a dictionary of sequences with their labels for training and test paths
        """
        paths = open( read_file, "r" )
        paths = paths.read().split( "\n" )
        paths_labels = dict()
        for item in paths:
            if item and item not in "":
                tools = item.split( "," )
                label = tools[ -1 ]
                train_tools = tools[ :len( tools ) - 1 ]
                train_tools = ",".join( train_tools )
                if train_tools in paths_labels:
                    paths_labels[ train_tools ] += "," + label
                else:
                    paths_labels[ train_tools ] = label
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
        label_mat = np.zeros( [ size_data, num_classes + 1 ] )
        train_counter = 0
        for train_seq, train_label in list( paths_dictionary.items() ):
            positions = train_seq.split( "," )
            start_pos = self.max_tool_sequence_len - len( positions )
            for id_pos, pos in enumerate( positions ):
                data_mat[ train_counter ][ start_pos + id_pos ] = int( pos )
            for label_item in train_label.split( "," ):
                label_mat[ train_counter ][ int( label_item ) ] = 1.0
            train_counter += 1
        return data_mat, label_mat

    @classmethod
    def get_filetype_compatibility( self, filetypes_path, dictionary ):
        """
        Get the next tools with compatible file types for each tool
        """
        with open( filetypes_path, "r" ) as compatible_tools_file:
            tools_compatibility = json.loads( compatible_tools_file.read() )
        return tools_compatibility

    @classmethod
    def get_data_labels_mat( self ):
        """
        Convert the training and test paths into corresponding numpy matrices
        """
        processed_data, raw_paths = self.process_processed_data( self.raw_file )
        dictionary, reverse_dictionary = self.create_data_dictionary( processed_data )
        num_classes = len( dictionary )
        # randomize all the paths before dividing them into train and test parts
        random.shuffle( raw_paths )
        # divide train and test paths
        test_share = self.test_share * len( raw_paths )
        test_paths = raw_paths[ :int( test_share ) ]
        train_paths = raw_paths[ int( test_share ): ]
        # process training and test paths in different ways
        self.decompose_paths( train_paths, dictionary, self.train_file, self.train_sequence_file )
        self.decompose_paths( test_paths, dictionary, self.test_file, self.test_sequence_file )
        self.take_actual_paths( test_paths, dictionary, self.test_actual_file, self.test_actual_sequence_file )
        # create sequences with labels for train and test paths
        train_paths_dict = self.prepare_paths_labels_dictionary( self.train_file, self.train_data_labels_dict )
        test_paths_dict = self.prepare_paths_labels_dictionary( self.test_file, self.test_data_labels_dict )
        test_actual_paths_dict = self.prepare_paths_labels_dictionary( self.test_actual_file, self.test_actual_data_labels_dict )
        # create 0 padded sequences from train and test paths
        train_data, train_labels = self.pad_paths( train_paths_dict, num_classes )
        test_data, test_labels = self.pad_paths( test_paths_dict, num_classes )
        test_actual_data, test_actual_labels = self.pad_paths( test_actual_paths_dict, num_classes )
        next_compatible_tools = self.get_filetype_compatibility( self.compatible_tools_filetypes, dictionary )
        return train_data, train_labels, test_data, test_labels, test_actual_data, test_actual_labels, dictionary, reverse_dictionary, next_compatible_tools
