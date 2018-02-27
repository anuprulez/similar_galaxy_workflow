"""
Predict nodes in graphichal data (Galaxy workflows) using Recurrent Neural Network (LSTM)
"""
import sys
import random
import collections
import time
import numpy as np


class PrepareData:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.raw_file = "data/workflow_steps.txt"
        self.train_file = "data/train_data.txt"
        self.label_file = "data/train_label.txt"

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
        reverse_dictionary = dict(zip( dictionary.values(), dictionary.keys() ) )
        return dictionary, reverse_dictionary
    
    @classmethod
    def create_train_labels_file( self, dictionary, raw_paths ):
        """
        Create training data with its labels with varying window sizes
        """
        test_data = list()
        len_dict = len( dictionary )
        train_data = list()
        train_label = list()
        print "preparing downstream data..."
        for index, item in enumerate( raw_paths ):
            tools = item.split(" ")
            max_window_size = len( tools )
            for window in range( 1, max_window_size ):
                slide_window_time = ( max_window_size - 1 ) // window
                for j in range( 0, slide_window_time ):
                    training_sequence = tools[ j: j + window ]
                    label = tools[ j + window: j + window + 1 ]
                    workflow_hot_vector_train = [ str( dictionary[ str( tool_item ) ] ) for tool_item in training_sequence ]
                    workflow_hot_vector_train = ",".join( workflow_hot_vector_train )
                    if workflow_hot_vector_train not in train_data:
                        train_data.append( workflow_hot_vector_train )
                        train_label.append( str( dictionary[ str( label[ 0 ] ) ] ) )
            print "Path %d processed" % ( index + 1 )
        with open( self.train_file, "w" ) as train_file:
            for item in train_data:
                train_file.write( "%s\n" % item )
        with open( self.train_label, "w" ) as label_file:
            for item in train_label:
                label_file.write( "%s\n" % item ) 
        print "Training data and labels files written"

    @classmethod
    def prepare_train_test_data( self ):
        """
        Read training data and its labels files
        """
        training_samples = list()
        training_labels = list()
        train_file = open( self.train_file, "r" )
        train_file = train_file.read().split( "\n" )
        for item in train_file:
            training_samples.append( item )

        label_file = open( self.label_file, "r" )
        label_file = label_file.read().split( "\n" )
        for item in label_file:
            training_labels.append( item )
        return training_samples, training_labels

    @classmethod
    def read_data( self ):
        """
        Convert the data into corresponding arrays
        """
        processed_data, raw_paths = self.process_processed_data( self.raw_file )
        dictionary, reverse_dictionary = self.create_data_dictionary( processed_data )
        #self.create_train_labels_file( dictionary, raw_paths )
        # all the nodes/tools are classes as well
        num_classes = len( dictionary )
        train_data_array = np.zeros([num_classes])
        train_data, train_labels = self.prepare_train_test_data()
        # initialize the training data matrix
        train_data_array = np.zeros( [ len( train_data ), num_classes ] )
        train_label_array = np.zeros( [ len( train_data ), num_classes ] )
        for index, item in enumerate( train_data ):
           positions = item.split( "," )
           for pos in positions:
               if pos:
                   train_data_array[ index ][ int( pos ) ] = 1.0
           pos_label = train_labels[ index ]
           if pos_label:
               train_label_array[ index ][ int( pos_label ) ] = 1.0
        return train_data_array, train_label_array
