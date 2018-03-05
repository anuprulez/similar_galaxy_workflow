"""
Predict nodes in graphichal data (Galaxy workflows) using Machine Learning
"""
import sys
import numpy as np
import random
import collections
import time
import math
import os
from random import shuffle

# machine learning library
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from keras.optimizers import RMSprop, Adam
from keras.callbacks import LambdaCallback

import prepare_data
import evaluate_top_results

class PredictNextTool:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.test_data_share = 0.3
        self.test_positions = list()
        self.current_working_dir = os.getcwd()
        self.sequence_file = self.current_working_dir + "/data/train_data_sequence.txt"
        self.network_config_json_path = self.current_working_dir + "/data/model.json"
        self.weights_path = self.current_working_dir + "/data/weights/trained_model.h5"
        self.loss_path = self.current_working_dir + "/data/loss_history.txt"
        self.accuracy_path = self.current_working_dir + "/data/accuracy_history.txt"
        self.epoch_weights_path = self.current_working_dir + "/data/weights/weights-epoch-{epoch:02d}.hdf5"

    @classmethod
    def divide_train_test_data( self ):
        """
        Divide data into train and test sets in a random way
        """
        data = prepare_data.PrepareData()
        complete_data, labels, dictionary, reverse_dictionary = data.read_data()
        complete_data = complete_data[ :len( complete_data ) - 1 ]
        labels = labels[ :len( labels ) - 1 ]
        len_data = len( complete_data )
        len_test_data = int( self.test_data_share * len_data )
        dimensions = len( complete_data[ 0 ] )
        # take random positions from the complete data to create test data
        data_indices = range( len_data )
        shuffle( data_indices )
        self.test_positions = data_indices[ :len_test_data ]
        train_positions = data_indices[ len_test_data: ]

        # create test and train data and labels
        train_data = np.zeros( [ len_data - len_test_data, dimensions ] )
        train_labels = np.zeros( [ len_data - len_test_data, dimensions ] )
        test_data = np.zeros( [ len_test_data, dimensions ] )
        test_labels = np.zeros( [ len_test_data, dimensions ] )

        for i, item in enumerate( train_positions ):
            train_data[ i ] = complete_data[ item ]
            train_labels[ i ] = labels[ item ]

        for i, item in enumerate( self.test_positions ):
            test_data[ i ] = complete_data[ item ]
            test_labels[ i ] = labels[ item ]
        return train_data, train_labels, test_data, test_labels, dimensions, dictionary, reverse_dictionary

    @classmethod
    def evaluate_LSTM_network( self ):
        """
        Create LSTM network and evaluate performance
        """
        print "Dividing data..."
        n_epochs = 30
        num_predictions = 5
        batch_size = 40
        dropout = 0.2
        train_data, train_labels, test_data, test_labels, dimensions, dictionary, reverse_dictionary = self.divide_train_test_data()
        # reshape train and test data
        train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
        train_labels = np.reshape(train_labels, (train_labels.shape[0], 1, train_labels.shape[1]))
        test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))
        test_labels = np.reshape(test_labels, (test_labels.shape[0], 1, test_labels.shape[1]))
        train_data_shape = train_data.shape
        optimizer = Adam(lr=0.0001)
        # define recurrent network
        model = Sequential()
        model.add( LSTM( 256, input_shape=( train_data_shape[ 1 ], train_data_shape[ 2 ] ), return_sequences=True ) )
        model.add( Dropout( dropout ) )
        #model.add( LSTM( 512, return_sequences=True ) )
        #model.add( Dropout( dropout ) )
        model.add( LSTM( 256, return_sequences=True ) )
        model.add( Dense( 256 ) )
        model.add( Dropout( dropout ) )
        model.add( Dense( dimensions ) )
        model.add( Activation( 'softmax' ) )
        model.compile( loss='categorical_crossentropy', optimizer=optimizer, metrics=[ 'accuracy' ] )

        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint( self.epoch_weights_path, verbose=1, mode='max' )
        #evaluate_each_epoch = LambdaCallback( on_epoch_end=evaluate_after_epoch )
        callbacks_list = [ checkpoint ]

        print "Start training..."
        model_fit_callbacks = model.fit( train_data, train_labels, epochs=n_epochs, batch_size=batch_size, callbacks=callbacks_list, shuffle=True )
        loss_values = model_fit_callbacks.history[ "loss" ]
        accuracy_values = model_fit_callbacks.history[ "acc" ]
        np_loss_values = np.array( loss_values )
        np_accuracy_values = np.array( accuracy_values )
        np.savetxt( self.loss_path, np_loss_values, delimiter="," )
        np.savetxt( self.accuracy_path, np_accuracy_values, delimiter="," )

        # save the network as json
        model_json = model.to_json()
        with open( self.network_config_json_path, "w" ) as json_file:
            json_file.write(model_json)
        # save the learned weights to h5 file
        model.save_weights( self.weights_path )

        print "Start predicting..."
        accuracy = model.evaluate( test_data, test_labels, verbose=0 )
        print "Loss: %.2f " % accuracy[ 0 ]
        print "Top-1 accuracy: %.2f " % accuracy[ 1 ]

        # get top n accuracy
        predict_tool = evaluate_top_results.EvaluateTopResults()
        predict_tool.evaluate_topn_epochs( n_epochs, num_predictions, dimensions, reverse_dictionary, test_data, test_labels )
        
    @classmethod
    def evaluate_after_epoch( self, epoch, logs ):
        """
        Evaluate performance after each epoch
        """
        print "Epoch evaluated..."

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
    def get_raw_paths( self ):
        """
        Read training data and its labels files
        """
        training_samples = list()
        training_labels = list()
        train_file = open( self.sequence_file, "r" )
        train_file = train_file.read().split( "\n" )
        for item in train_file:
            tools = item.split( "," )
            train_tools = tools[ :len( tools) - 1 ]
            train_tools = ",".join( train_tools )
            training_samples.append( train_tools )
            training_labels.append( tools[ -1 ] )
        return training_samples, training_labels
 
    @classmethod
    def see_predicted_tools( self, trained_model, test_data, dictionary, reverse_dictionary, dimensions ):
        """
        Use trained model to predict next tool
        """
        # predict random input sequences
        num_predict = len( test_data )
        num_predictions = 5
        train_data, train_labels = self.get_raw_paths()
        prediction_accuracy = self.get_top_predictions( num_predictions, test_data, train_labels, dimensions, trained_model, reverse_dictionary )
        print "No. total test inputs: %d" % num_predict
        print "No. correctly predicted: %d" % prediction_accuracy
        print "Prediction accuracy: %s" % str( float( prediction_accuracy ) / num_predict )
  

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_next_tool.py" )
        exit( 1 )
    start_time = time.time()
    predict_tool = PredictNextTool()
    predict_tool.evaluate_LSTM_network()
    end_time = time.time()
    print "Program finished in %s seconds" % str( end_time - start_time  )
