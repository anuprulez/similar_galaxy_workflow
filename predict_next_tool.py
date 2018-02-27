"""
Predict nodes in graphichal data (Galaxy workflows) using Machine Learning
"""
import sys
import numpy as np
import random
import collections
import time
import math
import matplotlib.pyplot as plt

# machine learning library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import prepare_data

class PredictNextTool:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.test_data_share = 0.2

    @classmethod
    def divide_train_test_data( self ):
        """
        Divide data into train and test sets in a random way
        """
        data = prepare_data.PrepareData()
        complete_data, labels = data.read_data()
        len_data = len( complete_data )
        dimensions = len( complete_data[ 0 ] )
        len_test_data = int( self.test_data_share * len_data )

        # initialize the train and test sets
        train_data = np.zeros( [ len_data - len_test_data, dimensions ] )
        train_labels = np.zeros( [ len_data - len_test_data, dimensions ] )
        test_data = np.zeros( [ len_test_data, dimensions ] )
        test_labels = np.zeros( [ len_test_data, dimensions ] )

        # take random positions from the complete data to create test data
        test_positions = random.sample( range( 0, len_data ), len_test_data )
        for index, pos in enumerate( test_positions ):
            test_data[ index ] = complete_data[ pos ]
            test_labels[ index ] = labels[ pos ]

        # take the remaining positions for train data
        train_positions = [ item for item in range( len_data ) if item not in test_positions ]
        for index, pos in enumerate( train_positions ):
            train_data[ index ] = complete_data[ pos ]
            train_labels[ index ] = labels[ pos ]
                
        print len(train_data)
        print len(train_labels)
        print len(test_data)
        print len(test_labels)

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_next_tool.py" )
        exit( 1 )
    predict_tool = PredictNextTool()
    predict_tool.divide_train_test_data()
