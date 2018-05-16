"""
Predict nodes in graphichal data (Galaxy workflows) using Machine Learning
"""
import sys
import numpy as np
import time
import os

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
        self.abs_top_pred_path = self.current_working_dir + "/data/abs_top_pred.txt"
        self.test_top_pred_path = self.current_working_dir + "/data/test_top_pred.txt"

    @classmethod
    def evaluate_LSTM_network( self ):
        """
        Create LSTM network and evaluate performance
        """
        print ( "Dividing data..." )
        n_epochs = 100
        batch_size = 40
        dropout = 0.5
        lstm_units = 128
        # get training and test data and their labels
        data = prepare_data.PrepareData()
        train_data, train_labels, test_data, test_labels, dictionary, reverse_dictionary = data.get_data_labels_mat()
        dimensions = len( dictionary )
        embedding_vec_size = 200
        optimizer = RMSprop( lr=0.01 )
        # define recurrent network
        model = Sequential()
        model.add( Embedding( dimensions, embedding_vec_size, mask_zero=True ) )
        model.add( SpatialDropout1D( dropout ) )
        model.add( LSTM( lstm_units, dropout=dropout, return_sequences=True, recurrent_dropout=dropout, activation='softsign' ) )
        model.add( LSTM( lstm_units, dropout=dropout, return_sequences=False, recurrent_dropout=dropout, activation='softsign' ) )
        model.add( Dense( dimensions, activation='sigmoid', activity_regularizer=regularizers.l2( 0.01 ) ) )
        model.compile( loss="binary_crossentropy", optimizer=optimizer )
        # save the network as json
        model_json = model.to_json()
        with open( self.network_config_json_path, "w" ) as json_file:
            json_file.write( model_json )
        model.summary()
        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint( self.epoch_weights_path, verbose=2, mode='max' )
        #predict_callback_train = PredictCallback( train_data, train_labels, n_epochs )
        predict_callback_test = PredictCallback( test_data, test_labels, n_epochs )
        callbacks_list = [ checkpoint, predict_callback_test ] #predict_callback_train
        print ( "Start training..." )
        model_fit_callbacks = model.fit( train_data, train_labels, validation_data=( test_data, test_labels ), batch_size=batch_size, epochs=n_epochs, callbacks=callbacks_list, shuffle=True )
        loss_values = model_fit_callbacks.history[ "loss" ]
        validation_loss = model_fit_callbacks.history[ "val_loss" ]
        np.savetxt( self.loss_path, np.array( loss_values ), delimiter="," )
        np.savetxt( self.val_loss_path, np.array( validation_loss ), delimiter="," )
        #np.savetxt( self.abs_top_pred_path, predict_callback_train.epochs_acc, delimiter="," )
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
