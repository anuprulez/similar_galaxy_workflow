"""
Predict nodes in graphichal data (Galaxy workflows) using Machine Learning
"""
import sys
import numpy as np
import random
import collections
import time
import math
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

import prepare_data

class PredictNextTool:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.test_data_share = 0.3
        self.test_positions = list()
        self.sequence_file = "data/train_data_sequence.txt"
        self.network_config_json_path = "data/model.json"
        self.weights_path = "data/weights/trained_model.h5"
        self.loss_path = "data/loss_history.txt"
        self.accuracy_path = "data/accuracy_history.txt"
        self.epoch_weights_path = "data/weights/weights-epoch-{epoch:02d}.hdf5"

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
        n_epochs = 500
        batch_size = 500
        dropout = 0.4
        train_data, train_labels, test_data, test_labels, dimensions, dictionary, reverse_dictionary = self.divide_train_test_data()
        # reshape train and test data
        train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
        train_labels = np.reshape(train_labels, (train_labels.shape[0], 1, train_labels.shape[1]))
        test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))
        test_labels = np.reshape(test_labels, (test_labels.shape[0], 1, test_labels.shape[1]))
        train_data_shape = train_data.shape

        # define recurrent network
        model = Sequential()
        model.add( LSTM( 256, input_shape=( train_data_shape[ 1 ], train_data_shape[ 2 ] ), return_sequences=True ) )
        #model.add( Dropout( dropout ) )
        model.add( LSTM( 512, return_sequences=True ) )
        #model.add( Dropout( dropout ) )
        model.add( LSTM( 256, return_sequences=True) )
        model.add( Dense( 256 ) )
        #model.add( Dropout( dropout ) )
        model.add( Dense( dimensions ) )
        model.add( Activation( 'softmax' ) )
        model.compile( loss='categorical_crossentropy', optimizer='rmsprop', metrics=[ 'accuracy' ] )

        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint( self.epoch_weights_path, verbose=1, mode='max' )
        callbacks_list = [ checkpoint ]     
        
        print "Start training..."
        model_fit_callbacks = model.fit( train_data, train_labels, epochs=n_epochs, batch_size=batch_size, callbacks=callbacks_list )
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
        '''print "Evaluating top n accuracy..."
        self.see_predicted_tools( model, test_data, dictionary, reverse_dictionary, dimensions )
        print "==============================="
        self.see_predicted_tools( self.load_saved_model( self.network_config_json_path, self.weights_path ), test_data, dictionary, reverse_dictionary, dimensions )
        print "================================="'''
        self.evaluate_topn_epochs( n_epochs, 5, dimensions, test_data, reverse_dictionary )

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

    @classmethod
    def evaluate_topn_epochs( self, n_epochs, num_predictions, dimensions, test_data, reverse_dictionary ):
        """
        Get topn accuracy over training epochs
        """
        topn_accuracy = list()
        train_data, train_labels = self.get_raw_paths()
        base_path = 'data/weights/weights-epoch-'
        for i in range( n_epochs ):
            ite = '0' + str( i + 1 ) if i < 9 else str( i + 1  )
            file_path = base_path + ite + '.hdf5'
            loaded_model = self.load_saved_model( self.network_config_json_path, file_path )
            accuracy = self.get_top_predictions( num_predictions, test_data, train_labels, dimensions, loaded_model, reverse_dictionary )
            topn_accuracy.append( accuracy )
        print topn_accuracy

    @classmethod
    def get_top_predictions( self, topn, test_data, train_labels, dimensions, trained_model, reverse_dictionary ):
        """
        Compute top n predictions with a trained model
        """
        print "Get top %d predictions for each test input..." % topn
        num_predict = len( self.test_positions )
        prediction_accuracy = 0
        for i in range( num_predict ):
            input_seq = test_data[ i ][ 0 ]
            label_text = train_labels[ self.test_positions[ i ] ]
            input_seq_reshaped = np.reshape( input_seq, ( 1, 1, dimensions ) )
            # predict the next tool using the trained model
            prediction = trained_model.predict( input_seq_reshaped, verbose=0 )
            prediction = np.reshape( prediction, ( dimensions ) )
            # take prediction in reverse order, best ones first
            prediction_pos = np.argsort( prediction, axis=0 )
            # get top n predictions
            top_prediction_pos = prediction_pos[ -topn: ]
            # get tool names for the predicted positions
            top_predictions = [ reverse_dictionary[ pred_pos ] for pred_pos in top_prediction_pos ]
            top_predicted_tools_text = " ".join( top_predictions )
            if label_text in top_predictions:
                prediction_accuracy += 1
            #print "Ordered input sequence: %s" % train_data[ self.test_positions[ i ] ]
            #print "Actual next tool: %s" % label_text
            #print "Predicted top %d next tools: %s" % ( num_predictions, top_predicted_tools_text )
            #print "=========================================="
        return float( prediction_accuracy ) / num_predict

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_next_tool.py" )
        exit( 1 )
    start_time = time.time()
    predict_tool = PredictNextTool()
    predict_tool.evaluate_LSTM_network()
    end_time = time.time()
    print "Program finished in %s seconds" % str( end_time - start_time  )
