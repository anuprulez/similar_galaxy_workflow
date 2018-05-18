"""
Predict nodes in graphichal data (Galaxy workflows) using Machine Learning
"""
import sys
import numpy as np
import time
import os
import json

# machine learning library
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, Callback
from keras import regularizers
from keras.layers.core import SpatialDropout1D
from keras.optimizers import RMSprop

import prepare_data


class PredictNextTool:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.network_config_json_path = self.current_working_dir + "/data/model.json"
        self.loss_path = self.current_working_dir + "/data/loss_history.txt"
        self.val_loss_path = self.current_working_dir + "/data/val_loss_history.txt"
        self.epoch_weights_path = self.current_working_dir + "/data/weights/weights-epoch-{epoch:02d}.hdf5"
        self.train_abs_top_pred_path = self.current_working_dir + "/data/train_abs_top_pred.txt"
        self.train_top_compatibility_pred_path = self.current_working_dir + "/data/train_top_compatible_pred.txt"
        self.test_abs_top_pred_path = self.current_working_dir + "/data/test_abs_top_pred.txt"
        self.test_top_compatibility_pred_path = self.current_working_dir + "/data/test_top_compatible_pred.txt"

    @classmethod
    def evaluate_LSTM_network( self, n_epochs=20, batch_size=40, dropout=0.0, lstm_units=64, embedding_vec_size=100, lr=0.01, reg_coeff=0.01 ):
        """
        Create LSTM network and evaluate performance
        """
        print ( "Dividing data..." )
        # get training and test data and their labels
        data = prepare_data.PrepareData()
        train_data, train_labels, test_data, test_labels, dictionary, reverse_dictionary, next_compatible_tools = data.get_data_labels_mat()
        # Increase the dimension by 1 to mask the 0th position
        dimensions = len( dictionary ) + 1
        optimizer = RMSprop( lr=lr )
        # define recurrent network
        model = Sequential()
        model.add( Embedding( dimensions, embedding_vec_size, mask_zero=True ) )
        model.add( SpatialDropout1D( 0.0 ) )
        model.add( LSTM( lstm_units, dropout=dropout, return_sequences=True, activation='softsign' ) )
        model.add( LSTM( lstm_units, dropout=dropout, return_sequences=False, activation='softsign' ) )
        model.add( Dense( dimensions, activation='sigmoid', activity_regularizer=regularizers.l2( reg_coeff ) ) )
        model.compile( loss="binary_crossentropy", optimizer=optimizer )
        # save the network as json
        model_json = model.to_json()
        with open( self.network_config_json_path, "w" ) as json_file:
            json_file.write( model_json )
        model.summary()
        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint( self.epoch_weights_path, verbose=2, mode='max' )
        #predict_callback_train = PredictCallback( train_data, train_labels, n_epochs, reverse_dictionary, next_compatible_tools )
        predict_callback_test = PredictCallback( test_data, test_labels, n_epochs, reverse_dictionary, next_compatible_tools )
        callbacks_list = [ checkpoint, predict_callback_test ] #predict_callback_train
        print ( "Start training..." )
        model_fit_callbacks = model.fit( train_data, train_labels, validation_data=( test_data, test_labels ), batch_size=batch_size, epochs=n_epochs, callbacks=callbacks_list, shuffle=True )
        loss_values = model_fit_callbacks.history[ "loss" ]
        validation_loss = model_fit_callbacks.history[ "val_loss" ]
        np.savetxt( self.loss_path, np.array( loss_values ), delimiter="," )
        np.savetxt( self.val_loss_path, np.array( validation_loss ), delimiter="," )
        #np.savetxt( self.train_abs_top_pred_path, predict_callback_train.abs_precision, delimiter="," )
        #np.savetxt( self.train_top_compatibility_pred_path, predict_callback_train.abs_compatible_precision, delimiter="," )
        np.savetxt( self.test_abs_top_pred_path, predict_callback_test.abs_precision, delimiter="," )
        np.savetxt( self.test_top_compatibility_pred_path, predict_callback_test.abs_compatible_precision, delimiter="," )
        print ( "Training finished" )


class PredictCallback( Callback ):
    def __init__( self, test_data, test_labels, n_epochs, reverse_data_dictionary, next_compatible_tools ):
        self.test_data = test_data
        self.test_labels = test_labels
        self.abs_precision = np.zeros( [ n_epochs ] )
        self.abs_compatible_precision = np.zeros( [ n_epochs ] )
        self.reverse_data_dictionary = reverse_data_dictionary
        self.next_compatible_tools = next_compatible_tools

    def on_epoch_end( self, epoch, logs={} ):
        """
        Compute topk accuracy for each test sample
        """
        x, y, reverse_data_dictionary, next_compatible_tools = self.test_data, self.test_labels, self.reverse_data_dictionary, self.next_compatible_tools
        size = y.shape[ 0 ]
        dimensions = y.shape[ 1 ]
        topk_abs_pred = np.zeros( [ size ] )
        topk_compatible_pred = np.zeros( [ size ] )
        # loop over all the test samples and find prediction precision
        for i in range( size ):
            correct_prediction_count = 0.0
            actual_classes_pos = np.where( y[ i ] > 0 )[ 0 ]
            topk = len( actual_classes_pos )
            test_sample = np.reshape( x[ i ], ( 1, x.shape[ 1 ] ) )
            test_sample_pos = np.where( x[ i ] > 0 )[ 0 ]
            test_sample_tool_pos = x[ i ][ test_sample_pos[ 0 ]: ]
            prediction = self.model.predict( test_sample, verbose=0 )
            prediction = np.reshape( prediction, ( dimensions, ) )
            prediction_pos = np.argsort( prediction, axis=-1 )
            topk_prediction_pos = prediction_pos[ -topk: ]
            # read tool names using reverse dictionary
            sequence_tool_names = [ reverse_data_dictionary[ int( tool_pos ) ] for tool_pos in test_sample_tool_pos ]
            actual_next_tool_names = [ reverse_data_dictionary[ int( tool_pos ) ] for tool_pos in actual_classes_pos ]
            top_predicted_next_tool_names = [ reverse_data_dictionary[ int( tool_pos ) ] for tool_pos in topk_prediction_pos ]
            # find false positives
            false_positives = [ tool_name for tool_name in top_predicted_next_tool_names if tool_name not in actual_next_tool_names ]
            absolute_precision = 1 - ( len( false_positives ) / float( len( actual_next_tool_names ) ) )
            adjusted_precision = absolute_precision
            # adjust the precision for compatible tools
            seq_last_tool = sequence_tool_names[ -1 ]
            if seq_last_tool in next_compatible_tools:
                next_tools = next_compatible_tools[ seq_last_tool ]
                next_tools = next_tools.split( "," )
                if len( next_tools ) > 0:
                    for false_pos in false_positives:
                        if false_pos in next_tools:
                            adjusted_precision += 1 / float( len( actual_next_tool_names ) )
            topk_abs_pred[ i ] = absolute_precision
            topk_compatible_pred[ i ] = adjusted_precision
        self.abs_precision[ epoch ] = np.mean( topk_abs_pred )
        self.abs_compatible_precision[ epoch ] = np.mean( topk_compatible_pred )
        print( "Epoch %d topk absolute precision: %.2f" % ( epoch + 1, np.mean( topk_abs_pred ) ) )
        print( "Epoch %d topk compatibility adjusted precision: %.2f" % ( epoch + 1, np.mean( topk_compatible_pred ) ) )


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_next_tool.py" )
        exit( 1 )
    start_time = time.time()
    predict_tool = PredictNextTool()
    predict_tool.evaluate_LSTM_network()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
