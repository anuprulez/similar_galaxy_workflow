"""
Predict nodes in graphichal data (Galaxy workflows) using Recurrent Neural Network (LSTM)
"""
import sys
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import random
import collections
import time

class PredictNextTool:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.train_data = "data/workflow_steps.txt"

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
    def create_train_test_data( self, dictionary, raw_paths ):
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
        with open( "data/train_data.txt", "w" ) as train_file:
            for item in train_data:
                train_file.write( "%s\n" % item )
        with open( "data/train_label.txt", "w" ) as label_file:
            for item in train_label:
                label_file.write( "%s\n" % item ) 
        print "Training data and labels files written"

    @classmethod
    def define_network( self ):
        print "create network"

    @classmethod
    def execute_network( self ):
        processed_data, raw_paths = self.process_processed_data( self.train_data )
        dictionary, reverse_dictionary = self.create_data_dictionary( processed_data )
        # all the nodes/tools are classes as well
        num_classes = len( dictionary )
        self.create_train_test_data( dictionary, raw_paths )


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_next_tool.py" )
        exit( 1 )
    predict_tool = PredictNextTool( )
    predict_tool.execute_network()



