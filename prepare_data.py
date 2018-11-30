"""
Prepare the workflow paths to be used by downstream
machine learning algorithm. The paths are divided
into the test and training sets
"""

import os
import collections
import numpy as np
import json
import random
import h5py

import utils


CURRENT_WORKING_DIR = os.getcwd()
RAW_FILE = CURRENT_WORKING_DIR + "/data/generated_files/workflow_connections_paths.txt"
DATA_DICTIONARY = CURRENT_WORKING_DIR + "/data/generated_files/data_dictionary.txt"
DATA_REV_DICT = CURRENT_WORKING_DIR + "/data/generated_files/data_rev_dict.txt"
COMPLETE_FILE = CURRENT_WORKING_DIR + "/data/generated_files/complete_file.txt"
COMPLETE_FILE_SEQUENCE = CURRENT_WORKING_DIR + "/data/generated_files/complete_file_sequence.txt"
COMPLETE_PATHS_POS = CURRENT_WORKING_DIR + "/data/generated_files/complete_paths_pos.txt"
COMPLETE_PATHS_NAMES = CURRENT_WORKING_DIR + "/data/generated_files/complete_paths_names.txt"
COMPLETE_PATHS_POS_DICT = CURRENT_WORKING_DIR + "/data/generated_files/complete_paths_pos_dict.json"
COMPLETE_PATHS_NAMES_DICT = CURRENT_WORKING_DIR + "/data/generated_files/complete_paths_names_dict.json"
TRAIN_DATA_LABELS_DICT = CURRENT_WORKING_DIR + "/data/generated_files/train_data_labels_names_dict.json"
TRAIN_DATA_LABELS_NAMES_DICT = CURRENT_WORKING_DIR + "/data/generated_files/train_data_labels_names_dict.json"
TEST_DATA_LABELS_DICT = CURRENT_WORKING_DIR + "/data/generated_files/test_data_labels_dict.json"
TEST_DATA_LABELS_NAMES_DICT = CURRENT_WORKING_DIR + "/data/generated_files/test_data_labels_names_dict.json"
TRAIN_DATA = CURRENT_WORKING_DIR + "/data/generated_files/train_data.h5"
TEST_DATA = CURRENT_WORKING_DIR + "/data/generated_files/test_data.h5"
TRAIN_DATA_CLASS_FREQ = CURRENT_WORKING_DIR + "/data/generated_files/train_data_class_freq.txt"


class PrepareData:

    @classmethod
    def __init__( self, max_seq_length, test_data_share, retrain=False ):
        """ Init method. """
        self.max_tool_sequence_len = max_seq_length
        self.test_share = test_data_share
        self.retrain = retrain

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
    def create_new_dict(self, new_data_dict):
        """
        Create new data dictionary
        """
        utils.remove_file(DATA_DICTIONARY)
        utils.remove_file(DATA_REV_DICT)
        reverse_dict = dict((v,k) for k,v in new_data_dict.items())
        utils.write_file(DATA_DICTIONARY, new_data_dict)
        utils.write_file(DATA_REV_DICT, reverse_dict)
        return new_data_dict, reverse_dict

    @classmethod
    def assemble_dictionary( self, new_data_dict):
        """
        Create/update tools indices in the forward and backward dictionary
        """
        if self.retrain is True or self.retrain is "True":
            with open( DATA_DICTIONARY, 'r' ) as data_dict:
                dictionary = json.loads( data_dict.read() )
                max_prev_size = len(dictionary)
                tool_counter = 1
                for tool in new_data_dict:
                    if tool not in dictionary:
                        dictionary[tool] = max_prev_size + tool_counter
                        tool_counter += 1
                reverse_dict = dict((v,k) for k,v in dictionary.items())
                utils.write_file(DATA_DICTIONARY, dictionary)
                utils.write_file(DATA_REV_DICT, reverse_dict)
            return dictionary, reverse_dict
        else:
            new_data_dict, reverse_dict = self.create_new_dict(new_data_dict)
            return new_data_dict, reverse_dict

    @classmethod
    def create_data_dictionary( self, words ):
        """
        Create two dictionaries having tools names and their indexes
        """
        count = collections.Counter( words ).most_common()
        dictionary = dict()
        for word, _ in count:
            dictionary[ word ] = len( dictionary ) + 1
        dictionary, reverse_dictionary = self.assemble_dictionary(dictionary)
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
        utils.remove_file(file_pos)
        utils.remove_file(file_names)
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
        utils.remove_file(paths_file_pos)
        utils.remove_file(paths_file_names)
        utils.remove_file(destination_file)
        utils.remove_file(destination_file_names)
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
        utils.remove_file(file_path)
        with open( file_path, "w" ) as dict_file:
            dict_file.write( json.dumps( dictionary ) )
        for item in dictionary:
            path_names = ",".join( [ reverse_dictionary[ int( pos ) ] for pos in item.split( "," ) ] )
            path_label_names = ",".join( [ reverse_dictionary[ int( pos ) ] for pos in dictionary[ item ].split( "," ) ] )
            path_seq_names[ path_names ] = path_label_names
        utils.remove_file(file_names_path)
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
    def save_as_h5py( self, data, label, file_path ):
        """
        Save the samples and their labels as h5 files
        """
        utils.remove_file(file_path)
        hf = h5py.File( file_path, 'w' )
        hf.create_dataset('data', data=data, compression="gzip", compression_opts=9)
        hf.create_dataset('data_labels', data=label, compression="gzip", compression_opts=9)
        hf.close()
        
    @classmethod
    def get_class_frequency(self, train_labels):
        """
        Compute class frequency and (inverse) class weights
        """
        n_classes = train_labels.shape[1]
        frequency_scores = dict()
        class_weights = list()
        class_weights.append(0)
        for i in range(1, n_classes):
            count = len(np.where( train_labels[:, i] > 0 )[0])
            frequency_scores[str(i)] = count
            class_weights.append(count)
        frequency_scores = dict(sorted(frequency_scores.items(), key=lambda kv: kv[1]))
        utils.write_file(TRAIN_DATA_CLASS_FREQ, frequency_scores)
        max_weight = max(class_weights)
        class_weights = [np.round((max_weight / float(wt)), 2) if wt > 0 else 0 for wt in class_weights]
        inverse_weights = np.asarray(class_weights, dtype=np.float64)
        return frequency_scores, inverse_weights

    @classmethod
    def get_data_labels_mat( self ):
        """
        Convert the training and test paths into corresponding numpy matrices
        """
        processed_data, raw_paths = self.process_processed_data( RAW_FILE )
        dictionary, reverse_dictionary = self.create_data_dictionary( processed_data )
        num_classes = len( dictionary )

        print( "Raw paths: %d" % len( raw_paths ) )
        random.shuffle( raw_paths )

        print( "Decomposing paths..." )
        all_unique_paths = self.decompose_paths( raw_paths, dictionary, COMPLETE_FILE, COMPLETE_FILE_SEQUENCE )
        random.shuffle( all_unique_paths )

        print( "Creating dictionaries..." )
        multilabels_paths = self.prepare_paths_labels_dictionary( reverse_dictionary, all_unique_paths, COMPLETE_PATHS_POS, COMPLETE_PATHS_NAMES, COMPLETE_PATHS_POS_DICT, COMPLETE_PATHS_NAMES_DICT )

        print( "Complete data: %d" % len( multilabels_paths ) )
        train_paths_dict, test_paths_dict = self.split_test_train_data( multilabels_paths )

        print( "Train data: %d" % len( train_paths_dict ) )
        print( "Test data: %d" % len( test_paths_dict ) )

        self.write_to_file( TEST_DATA_LABELS_DICT, TEST_DATA_LABELS_NAMES_DICT, test_paths_dict, reverse_dictionary )
        test_data, test_labels = self.pad_paths( test_paths_dict, num_classes )
        self.save_as_h5py( test_data, test_labels, TEST_DATA )

        self.write_to_file( TRAIN_DATA_LABELS_DICT, TRAIN_DATA_LABELS_NAMES_DICT, train_paths_dict, reverse_dictionary )
        train_data, train_labels = self.pad_paths( train_paths_dict, num_classes )
        train_data, train_labels = self.randomize_data( train_data, train_labels )
        frequency_scores, inverse_class_weights = self.get_class_frequency(train_labels)
        # get weighted class labels for each training sample
        weighted_train_labels = np.multiply(train_labels, inverse_class_weights)
        row_sums = weighted_train_labels.sum(axis=1)
        # normalize the weighted class values
        weighted_train_labels_normalised = weighted_train_labels / row_sums[:, np.newaxis]
        self.save_as_h5py( train_data, weighted_train_labels_normalised, TRAIN_DATA )
