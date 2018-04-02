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
from keras.callbacks import ModelCheckpoint, Callback
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from keras import backend as K
import tensorflow as tf

import prepare_data

class PredictNextTool:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.sequence_file = self.current_working_dir + "/data/train_data_sequence.txt"
        self.network_config_json_path = self.current_working_dir + "/data/model.json"
        self.loss_path = self.current_working_dir + "/data/loss_history.txt"
        self.val_loss_path = self.current_working_dir + "/data/val_loss_history.txt"
        self.epoch_weights_path = self.current_working_dir + "/data/weights/weights-epoch-{epoch:02d}.hdf5"
        self.test_data_path = self.current_working_dir + "/data/test_data.hdf5"
        self.test_labels_path = self.current_working_dir + "/data/test_labels.hdf5"
        self.abs_top_pred_path = self.current_working_dir + "/data/abs_top_pred.txt"
        self.test_top_pred_path = self.current_working_dir + "/data/test_top_pred.txt"

    @classmethod
    def divide_train_test_data( self ):
        """
        Divide data into train and test sets in a random way
        """
        test_data_share = 0.33
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
        return train_data, train_labels, test_data, test_labels, dimensions, dictionary, reverse_dictionary, complete_data, labels

    @classmethod
    def evaluate_LSTM_network( self ):
        """
        Create LSTM network and evaluate performance
        """
        print ( "Dividing data..." )
        n_epochs = 10
        batch_size = 60
        dropout = 0.5
        lstm_units = 256
        train_data, train_labels, test_data, test_labels, dimensions, dictionary, reverse_dictionary, comp_data, comp_labels = self.divide_train_test_data()
        embedding_vec_size = 100
        # define recurrent network
        model = Sequential()
        model.add( Embedding( dimensions, embedding_vec_size, mask_zero=True ) )
        model.add( LSTM( lstm_units, dropout=dropout, return_sequences=True, recurrent_dropout=dropout ) )
        model.add( LSTM( lstm_units, dropout=dropout, return_sequences=False, recurrent_dropout=dropout ) )
        model.add( Dense( dimensions, activation='sigmoid' ) )
        model.compile( loss="binary_crossentropy", optimizer='rmsprop' )
        # save the network as json
        model_json = model.to_json()
        with open( self.network_config_json_path, "w" ) as json_file:
            json_file.write( model_json )
        model.summary()
        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint( self.epoch_weights_path, verbose=2, mode='max' )
        predict_callback_complete = PredictCallback( comp_data, comp_labels, n_epochs )
        predict_callback_test = PredictCallback( test_data, test_labels, n_epochs )
        callbacks_list = [ checkpoint, predict_callback_complete, predict_callback_test ]
        print ( "Start training..." )
        model_fit_callbacks = model.fit( train_data, train_labels, validation_split=0.1, batch_size=batch_size, epochs=n_epochs, callbacks=callbacks_list, shuffle=True )
        loss_values = model_fit_callbacks.history[ "loss" ]
        validation_loss = model_fit_callbacks.history[ "val_loss" ]
        np.savetxt( self.loss_path, np.array( loss_values ), delimiter="," )
        np.savetxt( self.val_loss_path, np.array( validation_loss ), delimiter="," )
        np.savetxt( self.abs_top_pred_path, predict_callback_complete.epochs_acc, delimiter="," )
        np.savetxt( self.test_top_pred_path, predict_callback_test.epochs_acc, delimiter="," )
        print ( "Training finished" )


class PredictCallback( Callback ):
    def __init__( self, test_data, test_labels, n_epochs ):
        self.test_data = test_data
        self.test_labels = test_labels
        self.epochs_acc = np.zeros( [ n_epochs ] )

    def on_epoch_end( self, epoch, logs={} ):
        """
        Compute topk accuracy for each test sample
        """
        x, y = self.test_data, self.test_labels
        size = y.shape[ 0 ]
        dimensions = y.shape[ 1 ]
        topk_pred = np.zeros( [ size ] )
        for i in range( size ):
            correct_prediction_count = 0.0
            actual_classes_pos = np.where( y[ i ] > 0.0 )[ 0 ]
            topk = len( actual_classes_pos )
            test_sample = np.reshape( x[ i ], ( 1, x.shape[ 1 ] ) )
            prediction = self.model.predict( test_sample, verbose=0 )
            prediction = np.reshape( prediction, ( dimensions, ) )
            prediction_pos = np.argsort( prediction, axis=-1 )
            topk_prediction_pos = prediction_pos[ -topk: ]
            for item in topk_prediction_pos:
                if item in actual_classes_pos:
                    correct_prediction_count += 1.0
            topk_prediction_sample = correct_prediction_count / float( topk )
            topk_pred[ i ] = topk_prediction_sample
        epoch_mean_acc = np.mean( topk_pred )
        self.epochs_acc[ epoch ] = epoch_mean_acc
        print( "Epoch %d topk accuracy: %.2f" % ( epoch + 1, epoch_mean_acc ) )


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_next_tool.py" )
        exit( 1 )
    start_time = time.time()
    predict_tool = PredictNextTool()
    predict_tool.evaluate_LSTM_network()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
