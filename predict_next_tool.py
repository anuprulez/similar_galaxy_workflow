"""
Predict next tools in graphichal data (Galaxy workflows) using Machine Learning (Recurrent neural network)
"""

import sys
import numpy as np
import time
import os

# machine learning library
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import SpatialDropout1D
from keras.optimizers import RMSprop

import extract_workflow_connections
import prepare_data


class PredictNextTool:

    @classmethod
    def __init__( self, epochs ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.network_config_json_path = self.current_working_dir + "/data/model.json"
        self.epoch_weights_path = self.current_working_dir + "/data/weights/weights-epoch-{epoch:02d}.hdf5"
        self.mean_test_absolute_precision = self.current_working_dir + "/data/mean_test_absolute_precision.txt"
        self.mean_test_compatibility_precision = self.current_working_dir + "/data/mean_test_compatibility_precision.txt"
        self.mean_test_actual_absolute_precision = self.current_working_dir + "/data/mean_test_actual_absolute_precision.txt"
        self.mean_test_actual_compatibility_precision = self.current_working_dir + "/data/mean_test_actual_compatibility_precision.txt"
        self.mean_train_loss = self.current_working_dir + "/data/mean_train_loss.txt"
        self.mean_test_loss = self.current_working_dir + "/data/mean_test_loss.txt"
        self.n_epochs = epochs

    @classmethod
    def save_network( self, model ):
        """
        Save the network as json file
        """
        with open( self.network_config_json_path, "w" ) as json_file:
            json_file.write( model )

    @classmethod
    def evaluate_recurrent_network( self, run, network_config ):
        """
        Define recurrent neural network and train sequential data
        """
        print( "Experiment run: %d/%d" % ( ( run + 1 ), network_config[ "experiment_runs" ] ) )
        print ( "Dividing data..." )
        # get training and test data and their labels
        data = prepare_data.PrepareData( network_config[ "max_seq_len" ], network_config[ "test_share" ] )
        train_data, train_labels, test_data, test_labels, dictionary, reverse_dictionary, next_compatible_tools = data.get_data_labels_mat()
        # Increase the dimension by 1 to mask the 0th position
        dimensions = len( dictionary ) + 1
        optimizer = RMSprop( lr=network_config[ "learning_rate" ] )
        # define recurrent network
        model = Sequential()
        model.add( Embedding( dimensions, network_config[ "embedding_vec_size" ], mask_zero=True ) )
        model.add( SpatialDropout1D( network_config[ "dropout" ] ) )
        model.add( GRU( network_config[ "memory_units" ], dropout=network_config[ "dropout" ], recurrent_dropout=network_config[ "dropout" ], return_sequences=True, activation=network_config[ "activation_recurrent" ] ) )
        model.add( Dropout( network_config[ "dropout" ] ) )
        model.add( GRU( network_config[ "memory_units" ], dropout=network_config[ "dropout" ], recurrent_dropout=network_config[ "dropout" ], return_sequences=False, activation=network_config[ "activation_recurrent" ] ) )
        model.add( Dropout( network_config[ "dropout" ] ) )
        model.add( Dense( dimensions, activation=network_config[ "activation_output" ] ) )
        model.compile( loss=network_config[ "loss_type" ], optimizer=optimizer )
        # save the network as json
        self.save_network( model.to_json() )
        model.summary()
        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint( self.epoch_weights_path, verbose=2, mode='max' )
        #predict_callback_train = PredictCallback( train_data, train_labels, n_epochs, reverse_dictionary, next_compatible_tools )
        predict_callback_test = PredictCallback( test_data, test_labels, network_config[ "n_epochs" ], reverse_dictionary, next_compatible_tools )
        callbacks_list = [ checkpoint, predict_callback_test ] #predict_callback_train
        print ( "Start training..." )
        model_fit_callbacks = model.fit( train_data, train_labels, validation_data=( test_data, test_labels ), batch_size=network_config[ "batch_size" ], epochs=self.n_epochs, callbacks=callbacks_list, shuffle=True )
        loss_values = model_fit_callbacks.history[ "loss" ]
        validation_loss = model_fit_callbacks.history[ "val_loss" ]
        return {
            "train_loss": np.array( loss_values ),
            "test_loss": np.array( validation_loss ),
            "test_absolute_precision": predict_callback_test.abs_precision, 
            "test_compatibility_precision" : predict_callback_test.abs_compatible_precision
        }
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
            actual_classes_pos = np.where( y[ i ] > 0 )[ 0 ]
            topk = len( actual_classes_pos )
            test_sample = np.reshape( x[ i ], ( 1, x.shape[ 1 ] ) )
            test_sample_pos = np.where( x[ i ] > 0 )[ 0 ]
            test_sample_tool_pos = x[ i ][ test_sample_pos[ 0 ]: ]
            prediction = self.model.predict( test_sample, verbose=0 )
            prediction = np.reshape( prediction, ( dimensions, ) )
            # remove the 0th position as there is no tool at this index
            prediction = prediction[ 1: ]
            prediction_pos = np.argsort( prediction, axis=-1 )
            topk_prediction_pos = prediction_pos[ -topk: ]
            # read tool names using reverse dictionary
            sequence_tool_names = [ reverse_data_dictionary[ int( tool_pos ) ] for tool_pos in test_sample_tool_pos ]
            actual_next_tool_names = [ reverse_data_dictionary[ int( tool_pos ) ] for tool_pos in actual_classes_pos ]
            top_predicted_next_tool_names = [ reverse_data_dictionary[ int( tool_pos ) + 1 ] for tool_pos in topk_prediction_pos ]
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
        print( "-------" )


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python predict_next_tool.py" )
        exit( 1 )
    start_time = time.time()
    network_config = {
        "experiment_runs": 1,
        "n_epochs": 40,
        "batch_size": 128,
        "dropout": 0.3,
        "memory_units": 128,
        "embedding_vec_size": 128,
        "learning_rate": 0.001,
        "max_seq_len": 25,
        "test_share": 0.2,
        "validation_split": 0.2,
        "activation_recurrent": 'elu',
        "activation_output": 'sigmoid',
        "loss_type": "binary_crossentropy"
    }
    connections = extract_workflow_connections.ExtractWorkflowConnections()
    connections.read_tabular_file()
    n_epochs = network_config[ "n_epochs" ]
    experiment_runs = network_config[ "experiment_runs" ]
    predict_tool = PredictNextTool( n_epochs )
    test_abs_precision = np.zeros( [ experiment_runs, n_epochs ] )
    test_compatibility_precision = np.zeros( [ experiment_runs, n_epochs ] )
    test_actual_absolute_precision = np.zeros( [ experiment_runs, n_epochs ] )
    test_actual_compatibility_precision = np.zeros( [ experiment_runs, n_epochs ] )
    training_loss = np.zeros( [ experiment_runs, n_epochs ] )
    test_loss = np.zeros( [ experiment_runs, n_epochs ] )
    for run in range( experiment_runs ):
        results = predict_tool.evaluate_recurrent_network( run, network_config )
        test_abs_precision[ run ] = results[ "test_absolute_precision" ]
        test_compatibility_precision[ run ] = results[ "test_compatibility_precision" ]
        training_loss[ run ] = results[ "train_loss" ]
        test_loss[ run ] = results[ "test_loss" ]
    np.savetxt( predict_tool.mean_test_absolute_precision, np.mean( test_abs_precision, axis=0 ), delimiter="," )
    np.savetxt( predict_tool.mean_test_compatibility_precision, np.mean( test_compatibility_precision, axis=0 ), delimiter="," )
    np.savetxt( predict_tool.mean_train_loss, np.mean( training_loss, axis=0 ), delimiter="," )
    np.savetxt( predict_tool.mean_test_loss, np.mean( test_loss, axis=0 ), delimiter="," )
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
