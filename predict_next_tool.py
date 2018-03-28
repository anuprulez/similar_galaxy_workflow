"""
Predict nodes in graphichal data (Galaxy workflows) using Machine Learning
"""
import sys
import numpy as np
import time
import os
import h5py as h5

# machine learning library
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy
from keras import backend as K

import prepare_data


class PredictNextTool:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.sequence_file = self.current_working_dir + "/data/train_data_sequence.txt"
        self.network_config_json_path = self.current_working_dir + "/data/model.json"
        self.weights_path = self.current_working_dir + "/data/weights/trained_model.h5"
        self.loss_path = self.current_working_dir + "/data/loss_history.txt"
        self.accuracy_path = self.current_working_dir + "/data/accuracy_history.txt"
        self.val_loss_path = self.current_working_dir + "/data/val_loss_history.txt"
        self.val_accuracy_path = self.current_working_dir + "/data/val_accuracy_history.txt"
        self.epoch_weights_path = self.current_working_dir + "/data/weights/weights-epoch-{epoch:02d}.hdf5"
        self.test_data_path = self.current_working_dir + "/data/test_data.hdf5"
        self.test_labels_path = self.current_working_dir + "/data/test_labels.hdf5"

    @classmethod
    def divide_train_test_data( self ):
        """
        Divide data into train and test sets in a random way
        """
        test_data_share = 0.2
        seed = 0
        data = prepare_data.PrepareData()
        complete_data, labels, dictionary, reverse_dictionary = data.read_data()
        np.random.seed( seed )
        dimensions = len( dictionary )
        train_data, test_data, train_labels, test_labels = train_test_split( complete_data, labels, test_size=test_data_share, random_state=seed )
        # write the test data and labels to files for further evaluation
        with h5.File( self.test_data_path, "w" ) as test_data_file:
            test_data_file.create_dataset( "testdata", test_data.shape, data=test_data )
        with h5.File( self.test_labels_path, "w" ) as test_labels_file:
            test_labels_file.create_dataset( "testlabels", test_labels.shape, data=test_labels )
        return train_data, train_labels, test_data, test_labels, dimensions, dictionary, reverse_dictionary

    @classmethod
    def evaluate_LSTM_network( self ):
        """
        Create LSTM network and evaluate performance
        """
        print ("Dividing data...")
        n_epochs = 150
        batch_size = 40
        dropout = 0.5
        lstm_units = 256
        train_data, train_labels, test_data, test_labels, dimensions, dictionary, reverse_dictionary = self.divide_train_test_data()
        embedding_vec_size = 100
        # define recurrent network
        model = Sequential()
        model.add( Embedding( dimensions, embedding_vec_size, mask_zero=True ) )
        model.add( LSTM( lstm_units, dropout=dropout, return_sequences=True, recurrent_dropout=dropout ) )
        model.add( LSTM( lstm_units, dropout=dropout, return_sequences=False, recurrent_dropout=dropout ) )
        model.add( Dense( dimensions, activation='softmax' ) )
        model.compile( loss="binary_crossentropy", optimizer='rmsprop', metrics=[ categorical_accuracy ]  )
        # save the network as json
        model_json = model.to_json()
        with open( self.network_config_json_path, "w" ) as json_file:
            json_file.write( model_json )
        # save the learned weights to h5 file
        model.save_weights( self.weights_path )
        model.summary()
        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint( self.epoch_weights_path, verbose=2, mode='max' )
        callbacks_list = [ checkpoint ]
        print ("Start training...")
        model_fit_callbacks = model.fit( train_data, train_labels, validation_data=( test_data, test_labels ), batch_size=batch_size, epochs=n_epochs, callbacks=callbacks_list, shuffle=True )
        loss_values = model_fit_callbacks.history[ "loss" ]
        accuracy_values = model_fit_callbacks.history[ "categorical_accuracy" ]
        validation_loss = model_fit_callbacks.history[ "val_loss" ]
        validation_acc = model_fit_callbacks.history[ "val_categorical_accuracy" ]
        np.savetxt( self.loss_path, np.array( loss_values ), delimiter="," )
        np.savetxt( self.accuracy_path, np.array( accuracy_values ), delimiter="," )
        np.savetxt( self.val_loss_path, np.array( validation_loss ), delimiter="," )
        np.savetxt( self.val_accuracy_path, np.array( validation_acc ), delimiter="," )
        print ( "Training finished" )

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


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_next_tool.py" )
        exit( 1 )
    start_time = time.time()
    predict_tool = PredictNextTool()
    predict_tool.evaluate_LSTM_network()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
