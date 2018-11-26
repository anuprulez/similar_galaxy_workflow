"""
Predict next tools in the Galaxy workflows
using machine learning (recurrent neural network)
"""

import sys
import numpy as np
import time
import os
import json
import h5py

# machine learning library
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import SpatialDropout1D
from keras.optimizers import RMSprop
from keras.models import model_from_json

import extract_workflow_connections
import prepare_data


# file paths
CURRENT_WORKING_DIR = os.getcwd()
NETWORK_C0NFIG_JSON_PATH = CURRENT_WORKING_DIR + "/data/generated_files/model.json"
NEW_NETWORK_C0NFIG_JSON_PATH = CURRENT_WORKING_DIR + "/data/generated_files/new_model.json"
EPOCH_WEIGHTS_PATH = CURRENT_WORKING_DIR + "/data/weights/new_weights-epoch-{epoch:d}.hdf5"
DATA_REV_DICT = CURRENT_WORKING_DIR + "/data/generated_files/data_rev_dict.txt"
DATA_DICTIONARY = CURRENT_WORKING_DIR + "/data/generated_files/data_dictionary.txt"
TRAIN_DATA = CURRENT_WORKING_DIR + "/data/generated_files/train_data.h5"
TEST_DATA = CURRENT_WORKING_DIR + "/data/generated_files/test_data.h5"


class PredictNextTool:

    @classmethod
    def __init__( self, epochs, trained_model_path ):
        """ Init method. """
        self.n_epochs = epochs
        self.TRAINED_MODEL_PATH = trained_model_path
        self.BEST_RETRAINED_MODEL_PATH = CURRENT_WORKING_DIR + "/data/weights/new_weights-epoch-" + str(epochs) + ".hdf5"

    @classmethod
    def save_network( self, model ):
        """
        Save the network as json file
        """
        with open( NETWORK_C0NFIG_JSON_PATH, "w" ) as json_file:
            json_file.write( model )

    @classmethod
    def read_file( self, file_path ):
        """
        Read a file
        """
        with open( file_path, "r" ) as json_file:
            file_content = json.loads( json_file.read() )
        return file_content

    @classmethod
    def get_h5_data( self, file_name ):
        """
        Read h5 file to get train and test data
        """
        hf = h5py.File( file_name, 'r' )
        return hf.get( "data" ), hf.get( "data_labels" )
        
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
    def save_network( self, model ):
        """
        Save the network as json file
        """
        with open( NETWORK_C0NFIG_JSON_PATH, "w" ) as json_file:
            json_file.write( model )

    @classmethod
    def verify_model(self, model, x, y, reverse_data_dictionary, dimensions):
        """
        Verify the model on test data
        """
        size = y.shape[ 0 ]
        topk_abs_pred = np.zeros( [ size ] )
        topk_compatible_pred = np.zeros( [ size ] )
        ave_abs_precision = list()
        # loop over all the test samples and find prediction precision
        for i in range( size ):
            actual_classes_pos = np.where( y[ i ] > 0 )[ 0 ]
            topk = len( actual_classes_pos )
            test_sample = np.reshape( x[ i ], ( 1, x.shape[ 1 ] ) )
            test_sample_pos = np.where( x[ i ] > 0 )[ 0 ]
            test_sample_tool_pos = x[ i ][ test_sample_pos[ 0 ]: ]

            # predict next tools for a test path
            prediction = model.predict( test_sample, verbose=0 )

            # remove the 0th position as there is no tool at this index
            prediction = np.reshape(prediction, (dimensions,))
            prediction = prediction[ 1:dimensions ]
            prediction_pos = np.argsort( prediction, axis=-1 )
            topk_prediction_pos = prediction_pos[ -topk: ]

            # read tool names using reverse dictionary
            sequence_tool_names = [ reverse_data_dictionary[ str( int( tool_pos ) ) ] for tool_pos in test_sample_tool_pos ]
            actual_next_tool_names = [ reverse_data_dictionary[ str( int( tool_pos ) ) ] for tool_pos in actual_classes_pos ]
            top_predicted_next_tool_names = [ reverse_data_dictionary[ str( int( tool_pos ) + 1 ) ] for tool_pos in topk_prediction_pos ]

            # find false positives
            false_positives = [ tool_name for tool_name in top_predicted_next_tool_names if tool_name not in actual_next_tool_names ]
            absolute_precision = 1 - ( len( false_positives ) / float( len( actual_next_tool_names ) ) )
            ave_abs_precision.append(absolute_precision)
        return np.mean(ave_abs_precision)
        
    @classmethod
    def retrain_model(self, training_data, training_labels, test_data, test_labels, network_config):
        """
        Retrain the trained model with new data and compare performance on test data
        """
        # retrain model
        print("New training size: %d" % len(training_labels))
        loaded_model = predict_tool.load_saved_model( NETWORK_C0NFIG_JSON_PATH, self.TRAINED_MODEL_PATH )
        print("Old model summary: \n")
        print(loaded_model.summary())
        old_dimensions = test_labels.shape[1]
        new_dimensions = training_labels.shape[1]
        model = Sequential()
        # add embedding
        model.add( Embedding( new_dimensions, network_config[ "embedding_vec_size" ], mask_zero=True ) )     
        new_embedding_dimensions = np.zeros([new_dimensions, network_config[ "embedding_vec_size" ]])
        new_embedding_dimensions[0:old_dimensions,:] = loaded_model.layers[0].get_weights()[0]       

        model.layers[0].set_weights([new_embedding_dimensions])
        model.layers[0].trainable = True
        model.add( SpatialDropout1D( network_config[ "dropout" ] ) )

        # add GRU
        model.add( GRU( network_config[ "memory_units" ], dropout=network_config[ "dropout" ], recurrent_dropout=network_config[ "dropout" ], return_sequences=True, activation=network_config[ "activation_recurrent" ] ) )
        model.layers[2].set_weights(loaded_model.layers[2].get_weights())
        model.layers[2].trainable = True
        model.add( Dropout( network_config[ "dropout" ] ) )

        # add GRU
        model.add( GRU( network_config[ "memory_units" ], dropout=network_config[ "dropout" ], recurrent_dropout=network_config[ "dropout" ], return_sequences=False, activation=network_config[ "activation_recurrent" ] ) )
        model.layers[4].set_weights(loaded_model.layers[4].get_weights())
        model.layers[4].trainable = True
        model.add( Dropout( network_config[ "dropout" ] ) )

        new_output_dimensions1 = np.zeros([network_config[ "memory_units" ], new_dimensions])
        new_output_dimensions1[:, 0:old_dimensions] = loaded_model.layers[6].get_weights()[0]
        new_output_dimensions2 = np.zeros([new_dimensions])
        new_output_dimensions2[:old_dimensions] = loaded_model.layers[6].get_weights()[1]

        model.add( Dense(new_dimensions, activation=network_config[ "activation_output" ]))
        model.layers[6].set_weights([new_output_dimensions1, new_output_dimensions2])
        model.layers[6].trainable = True
        optimizer = RMSprop( lr=network_config[ "learning_rate" ] )
        model.compile( loss=network_config[ "loss_type" ], optimizer=optimizer )
        
        # save the network as json
        self.save_network( model.to_json() )
        model.summary()
        
        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint( EPOCH_WEIGHTS_PATH, verbose=0, mode='max' )
        callbacks_list = [ checkpoint ]
        
        print("New model summary: \n")
        print(model.summary())

        reshaped_test_labels = np.zeros([test_labels.shape[0], new_dimensions])
        print("Started training on new data...")
        model_fit_callbacks = model.fit(training_data, training_labels, shuffle="batch", batch_size=network_config[ "batch_size" ], epochs=self.n_epochs, callbacks=callbacks_list)
        print ( "Training finished" )


if __name__ == "__main__":

    if len(sys.argv) != 4:
        print( "Usage: python retrain_model.py <workflow_file_path> <training_epochs> <trained_model_path>" )
        exit( 1 )
    start_time = time.time()
    
    retrain = True
    n_epochs = int(sys.argv[2])
    
    network_config = {
        "experiment_runs": 1,
        "n_epochs": n_epochs,
        "batch_size": 128,
        "dropout": 0.2,
        "memory_units": 128,
        "embedding_vec_size": 128,
        "learning_rate": 0.001,
        "max_seq_len": 25,
        "test_share": 0.20,
        "validation_split": 0.2,
        "activation_recurrent": 'elu',
        "activation_output": 'sigmoid',
        "loss_type": "binary_crossentropy"
    }

    experiment_runs = network_config[ "experiment_runs" ]

    # Extract and process workflows
    connections = extract_workflow_connections.ExtractWorkflowConnections(sys.argv[1], retrain)
    connections.read_tabular_file()

    # Process the paths from workflows
    print ( "Dividing data..." )
    data = prepare_data.PrepareData( network_config[ "max_seq_len" ], network_config[ "test_share" ], retrain)
    data.get_data_labels_mat()

    predict_tool = PredictNextTool( n_epochs, sys.argv[3] )
    # get data dictionary
    data_dict = predict_tool.read_file( DATA_DICTIONARY )
    reverse_data_dictionary = predict_tool.read_file( DATA_REV_DICT )

    # get training and test data with their labels
    train_data, train_labels = predict_tool.get_h5_data( TRAIN_DATA )
    test_data, test_labels = predict_tool.get_h5_data( TEST_DATA )

    # execute experiment runs and collect results for each run
    predict_tool.retrain_model( train_data, train_labels, test_data, test_labels, network_config )
        
    print("Evaluating performance on test data...")
    print("Test data size: %d" % len(test_labels))
    loaded_model = predict_tool.load_saved_model( NETWORK_C0NFIG_JSON_PATH, predict_tool.BEST_RETRAINED_MODEL_PATH )
    absolute_prec_current_model = predict_tool.verify_model(loaded_model, test_data, test_labels, reverse_data_dictionary, test_labels.shape[1])
    print("Absolute precision on test data using new model is: %0.6f" % absolute_prec_current_model)

    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
