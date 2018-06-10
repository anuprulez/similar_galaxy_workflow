"""
Prepare the workflows to be used by downstream machine learning algorithms
"""

import os
import collections
import numpy as np
import json
import random
import h5py


class PrepareData:

    @classmethod
    def __init__( self, max_seq_length, test_data_share ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.raw_file = self.current_working_dir + "/data/workflow_connections_paths.txt"
        self.data_dictionary = self.current_working_dir + "/data/data_dictionary.txt"
        self.data_rev_dict = self.current_working_dir + "/data/data_rev_dict.txt"
        self.complete_file = self.current_working_dir + "/data/complete_file.txt"
        self.complete_file_sequence = self.current_working_dir + "/data/complete_file_sequence.txt"
        self.complete_paths_pos = self.current_working_dir + "/data/complete_paths_pos.txt"
        self.complete_paths_names = self.current_working_dir + "/data/complete_paths_names.txt"
        self.complete_paths_pos_dict = self.current_working_dir + "/data/complete_paths_pos_dict.json"
        self.complete_paths_names_dict = self.current_working_dir + "/data/complete_paths_names_dict.json"
        self.train_data_labels_dict = self.current_working_dir + "/data/train_data_labels_dict.json"
        self.train_data_labels_names_dict = self.current_working_dir + "/data/train_data_labels_names_dict.json"
        self.test_data_labels_dict = self.current_working_dir + "/data/test_data_labels_dict.json"
        self.test_data_labels_names_dict = self.current_working_dir + "/data/test_data_labels_names_dict.json"
        self.compatible_tools_filetypes = self.current_working_dir + "/data/compatible_tools.json"
        self.paths_frequency = self.current_working_dir + "/data/workflow_paths_freq.txt"
        self.train_data = self.current_working_dir + "/data/train_data.h5"
        self.test_data = self.current_working_dir + "/data/test_data.h5"
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
                if token is not "":
                    tokens.append( token )
        tokens = list( set( tokens ) )
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
                        sub_paths_pos.append( ",".join( tools_pos ) )
                        sub_paths_names.append(  ",".join( sequence ) )
        sub_paths_pos = list( set( sub_paths_pos ) )
        sub_paths_names = list( set( sub_paths_names ) )
        with open( file_pos, "w" ) as sub_paths_file_pos:
            for item in sub_paths_pos:
                sub_paths_file_pos.write( "%s\n" % item )
        with open( file_names, "w" ) as sub_paths_file_names:
            for item in sub_paths_names:
                sub_paths_file_names.write( "%s\n" % item )
        return sub_paths_pos

    @classmethod
    def prepare_paths_labels_dictionary( self, reverse_dictionary, paths, paths_file_pos, paths_file_names, destination_file, destination_file_names ):
        """
        Create a dictionary of sequences with their labels for training and test paths
        """
        paths_labels = dict()
        paths_labels_names = dict()
        random.shuffle( paths )
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
        with open( paths_file_pos, "w" ) as write_paths_file_pos:
            for item in paths:
                write_paths_file_pos.write( "%s\n" % item )
        with open( paths_file_names, "w" ) as write_paths_file_names:
            for item in paths:
                write_paths_file_names.write( "%s\n" % ",".join( [ reverse_dictionary[ int( pos ) ] for pos in item.split( "," ) ] ) )
        with open( destination_file, 'w' ) as multilabel_file:
            multilabel_file.write( json.dumps( paths_labels ) )
        for item in paths_labels:
            path_names = ",".join( [ reverse_dictionary[ int( pos ) ] for pos in item.split( "," ) ] )
            path_label_names = ",".join( [ reverse_dictionary[ int( pos ) ] for pos in paths_labels[ item ].split( "," ) ] )
            paths_labels_names[ path_names ] = path_label_names
        with open( destination_file_names, "w" ) as multilabel_file_names:
            multilabel_file_names.write( json.dumps( paths_labels_names ) )
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
    def write_to_file( self, file_path, file_names_path, dictionary, reverse_dictionary ):
        """
        Write to file
        """
        path_seq_names = dict()
        with open( file_path, "w" ) as dict_file:
            dict_file.write( json.dumps( dictionary ) )
        for item in dictionary:
            path_names = ",".join( [ reverse_dictionary[ int( pos ) ] for pos in item.split( "," ) ] )
            path_label_names = ",".join( [ reverse_dictionary[ int( pos ) ] for pos in dictionary[ item ].split( "," ) ] )
            path_seq_names[ path_names ] = path_label_names
        with open( file_names_path, "w" ) as multilabel_file:
            multilabel_file.write( json.dumps( path_seq_names ) )

    @classmethod
    def split_test_train_data( self, multilabels_paths ):
        """
        Split into test and train data randomly for each run
        """
        train_dict = dict()
        test_dict = dict()
        all_paths = multilabels_paths.keys()
        random.shuffle( list( all_paths ) )
        split_number = int( self.test_share * len( all_paths ) )
        for index, path in enumerate( list( all_paths ) ):
            if index < split_number:
                test_dict[ path ] = multilabels_paths[ path ]
            else:
                train_dict[ path ] = multilabels_paths[ path ]
        return train_dict, test_dict

    @classmethod
    def randomize_data( self, train_data, train_labels ):
        """
        Randomize the train data after its inflation
        """
        size_data = train_data.shape
        size_labels = train_labels.shape
        rand_train_data = np.zeros( [ size_data[ 0 ], size_data[ 1 ] ] )
        rand_train_labels = np.zeros( [ size_labels[ 0 ], size_labels[ 1 ] ] )
        indices = np.arange( size_data[ 0 ] )
        random.shuffle( indices )
        for index, random_index in enumerate( indices ):
            rand_train_data[ index ] = train_data[ random_index ]
            rand_train_labels[ index ] = train_labels[ random_index ]
        return rand_train_data, rand_train_labels

    @classmethod
    def reconstruct_original_distribution( self, reverse_dictionary, train_data, train_labels ):
        """
        Reconstruct the original distribution in training data
        """
        paths_frequency = dict()
        repeated_train_sample = None
        repeated_train_sample_label = None
        train_data_size = train_data.shape[ 0 ]
        with open( self.paths_frequency, "r" ) as frequency:
            paths_frequency = json.loads( frequency.read() )
        for i in range( train_data_size ):
            label_tool_pos = np.where( train_labels[ i ] > 0 )[ 0 ]
            train_sample = np.reshape( train_data[ i ], ( 1, train_data.shape[ 1 ] ) )
            train_sample_pos = np.where( train_data[ i ] > 0 )[ 0 ]
            train_sample_tool_pos = train_data[ i ][ train_sample_pos[ 0 ]: ]
            sample_tool_names = ",".join( [ reverse_dictionary[ int( tool_pos ) ] for tool_pos in train_sample_tool_pos ] )
            label_tool_names = [ reverse_dictionary[ int( tool_pos ) ] for tool_pos in label_tool_pos ]
            for label in label_tool_names:
                reconstructed_path = sample_tool_names + "," + label
                if reconstructed_path in paths_frequency:
                    # subtract by one
                    adjusted_freq = paths_frequency[ reconstructed_path ] - 1
                    tr_data = np.tile( train_data[ i ], ( adjusted_freq, 1 ) )
                    tr_label = np.tile( train_labels[ i ], ( adjusted_freq, 1 ) )
                    if repeated_train_sample is not None: 
                        repeated_train_sample = np.vstack( ( repeated_train_sample, tr_data  ) )
                        repeated_train_sample_label = np.vstack( ( repeated_train_sample_label, tr_label  ) )
                    else:
                        repeated_train_sample = tr_data
                        repeated_train_sample_label = tr_label
            print( "Path reconstructed: %d" % i )
        train_data = np.vstack( ( train_data, repeated_train_sample ) )
        train_labels = np.vstack( ( train_labels, repeated_train_sample_label ) )
        return train_data, train_labels

    @classmethod
    def verify_overlap( self, train_data, test_data, reverse_dictionary ):
        """
        Verify the overlapping of samples in train and test data
        """
        train_data_size = train_data.shape[ 0 ]
        test_data_size = test_data.shape[ 0 ]
        train_samples = list()
        test_samples = list()
        for i in range( train_data_size ):
            train_sample = np.reshape( train_data[ i ], ( 1, train_data.shape[ 1 ] ) )
            train_sample_pos = np.where( train_data[ i ] > 0 )[ 0 ]
            train_sample_tool_pos = train_data[ i ][ train_sample_pos[ 0 ]: ]
            sample_tool_names = ",".join( [ str(tool_pos) for tool_pos in train_sample_tool_pos ] )
            train_samples.append( sample_tool_names )
        for i in range( test_data_size ):
            test_sample = np.reshape( test_data[ i ], ( 1, test_data.shape[ 1 ] ) )
            test_sample_pos = np.where( test_data[ i ] > 0 )[ 0 ]
            test_sample_tool_pos = test_data[ i ][ test_sample_pos[ 0 ]: ]
            sample_tool_names = ",".join( [ str(tool_pos) for tool_pos in test_sample_tool_pos ] )
            test_samples.append( sample_tool_names )
        intersection = list( set( train_samples ).intersection( set( test_samples ) ) )
        print( "Overlap in train and test: %d" % len( intersection ) )
        
    @classmethod
    def save_as_h5py( self, data, label, file_path ):
        """
        Save the samples and their labels as h5 files
        """
        hf = h5py.File( file_path, 'w' )
        hf.create_dataset( 'data', data=data, compression="gzip", compression_opts=9 )
        hf.create_dataset( 'data_labels', data=label, compression="gzip", compression_opts=9 )
        hf.close()

    @classmethod
    def get_data_labels_mat( self ):
        """
        Convert the training and test paths into corresponding numpy matrices
        """
        processed_data, raw_paths = self.process_processed_data( self.raw_file )
        dictionary, reverse_dictionary = self.create_data_dictionary( processed_data )
        num_classes = len( dictionary )
        print( "Raw paths: %d" % len( raw_paths ) )
        random.shuffle( raw_paths )

        print( "Decomposing paths..." )
        all_unique_paths = self.decompose_paths( raw_paths, dictionary, self.complete_file, self.complete_file_sequence )
        random.shuffle( all_unique_paths )

        print( "Creating dictionaries..." )
        multilabels_paths = self.prepare_paths_labels_dictionary( reverse_dictionary, all_unique_paths, self.complete_paths_pos, self.complete_paths_names, self.complete_paths_pos_dict, self.complete_paths_names_dict )

        print( "Complete data: %d" % len( multilabels_paths ) )
        train_paths_dict, test_paths_dict = self.split_test_train_data( multilabels_paths )

        print( "Train data: %d" % len( train_paths_dict ) )
        print( "Test data: %d" % len( test_paths_dict ) )
        self.write_to_file( self.test_data_labels_dict, self.test_data_labels_names_dict, test_paths_dict, reverse_dictionary )
        self.write_to_file( self.train_data_labels_dict, self.train_data_labels_names_dict, train_paths_dict, reverse_dictionary )

        print( "Padding paths with 0s..." )
        test_data, test_labels = self.pad_paths( test_paths_dict, num_classes )
        train_data, train_labels = self.pad_paths( train_paths_dict, num_classes )

        print( "Verifying overlap in train and test data..." )
        self.verify_overlap( train_data, test_data, reverse_dictionary )

        print( "Restoring the original data distribution in training data..." )
        train_data, train_labels = self.reconstruct_original_distribution( reverse_dictionary, train_data, train_labels )

        print( "Randomizing the train data..." )
        train_data, train_labels = self.randomize_data( train_data, train_labels )
        
        self.save_as_h5py( train_data, train_labels, self.train_data )
        self.save_as_h5py( test_data, test_labels, self.test_data )
