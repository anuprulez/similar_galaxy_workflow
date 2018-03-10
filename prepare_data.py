"""
Predict nodes in graphichal data (Galaxy workflows) using Recurrent Neural Network (LSTM)
"""

import os
import collections
import numpy as np
import json
import gensim
from gensim.models.doc2vec import TaggedDocument


class PrepareData:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.raw_file = self.current_working_dir + "/data/workflow_steps.txt"
        self.train_file = self.current_working_dir + "/data/train_data.txt"
        self.sequence_file = self.current_working_dir + "/data/train_data_sequence.txt"
        self.distribution_file = self.current_working_dir + "/data/data_distribution.txt"
        self.data_dictionary = self.current_working_dir + "/data/data_dictionary.txt"
        self.data_rev_dict = self.current_working_dir + "/data/data_rev_dict.txt"

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
            dictionary[word] = len( dictionary )
        reverse_dictionary = dict( zip( dictionary.values(), dictionary.keys() ) )
        with open( self.data_dictionary, 'w' ) as distribution_file:
            distribution_file.write( json.dumps( dictionary ) )
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
            max_window_size = len( tools )
            for window in range( 1, max_window_size ):
                slide_window_time = ( max_window_size - 1 ) // window
                for j in range( 0, slide_window_time ):
                    training_sequence = tools[ j: j + window ]
                    label = tools[ j + window: j + window + 1 ]
                    data_seq = ",".join( training_sequence )
                    data_seq += "," + label[ 0 ]

                    tools_pos = [ str( dictionary[ str( tool_item ) ] ) for tool_item in training_sequence ]
                    tools_pos = ",".join( tools_pos )
                    tools_pos = tools_pos + "," + str( dictionary[ str( label[ 0 ] ) ] )

                    if tools_pos not in train_data:
                        train_data.append( tools_pos )

                    if data_seq not in train_data_sequence:
                        train_data_sequence.append( data_seq )

            print ( "Path %d processed" % ( index + 1 ) )

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
        training_samples = list()
        training_labels = list()
        train_file = open( self.train_file, "r" )
        train_file = train_file.read().split( "\n" )
        seq_len = list()
        data_distribution = dict()
        for item in train_file:
            tools = item.split( "," )
            train_tools = tools[ :len( tools) - 1 ]
            train_tools = ",".join( train_tools )
            label = tools[ -1 ]
            if label:
                if label in data_distribution:
                    data_distribution[ label ] += 1
                else:
                    data_distribution[ label ] = 1
                seq_len.append( len( train_tools ) )
                training_labels.append( tools[ -1 ] )
                training_samples.append( train_tools )
        # save the data distribution - count the number of samples with same class
        with open( self.distribution_file, 'w' ) as distribution_file:
            distribution_file.write( json.dumps( data_distribution ) )
        return training_samples, training_labels, max( seq_len )

    @classmethod
    def read_data( self ):
        """
        Convert the data into corresponding arrays
        """
        tagged_documents = list()
        processed_data, raw_paths = self.process_processed_data( self.raw_file )
        dictionary, reverse_dictionary = self.create_data_dictionary( processed_data )
        self.create_train_labels_file( dictionary, raw_paths )
        # all the nodes/tools are classes as well
        num_classes = len( dictionary )
        train_data, train_labels, max_seq_len = self.prepare_train_test_data()
        # initialize the training data matrix
        train_data_array = np.zeros( [ len( train_data ), max_seq_len ] )
        train_label_array = np.zeros( [ len( train_data ), num_classes ] )
        for index, item in enumerate( train_data ):
            positions = item.split( "," )
            nodes = list()
            for id_pos, pos in enumerate( positions ):
                if pos:
                    train_data_array[ index ][ id_pos ] = int( pos )
                    nodes.append( reverse_dictionary[ int( pos ) ] )
            # tagged documents
            td = TaggedDocument( gensim.utils.to_unicode( " ".join( nodes ) ), [ index ] )
            tagged_documents.append( td )
            pos_label = train_labels[ index ]
            if pos_label:
                # one-hot vector for labels
                train_label_array[ index ][ int( pos_label ) ] = 1.0
        return train_data_array, train_label_array, dictionary, reverse_dictionary, tagged_documents
