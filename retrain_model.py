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
import utils


# file paths
CURRENT_WORKING_DIR = os.getcwd()
NETWORK_C0NFIG_JSON_PATH = CURRENT_WORKING_DIR + "/data/generated_files/model.json"
#NEW_NETWORK_C0NFIG_JSON_PATH = CURRENT_WORKING_DIR + "/data/generated_files/new_model.json"
EPOCH_WEIGHTS_PATH = CURRENT_WORKING_DIR + "/data/weights/weights-epoch-{epoch:d}.hdf5"
DATA_REV_DICT = CURRENT_WORKING_DIR + "/data/generated_files/data_rev_dict.txt"
DATA_DICTIONARY = CURRENT_WORKING_DIR + "/data/generated_files/data_dictionary.txt"
TRAIN_DATA = CURRENT_WORKING_DIR + "/data/generated_files/train_data.h5"
TEST_DATA = CURRENT_WORKING_DIR + "/data/generated_files/test_data.h5"
#TEST_DATA_NEW = CURRENT_WORKING_DIR + "/data/generated_files/test_data_new.h5"


class PredictNextTool:

    @classmethod
    def __init__( self, epochs, trained_model_path ):
        """ Init method. """
        self.n_epochs = epochs
        self.TRAINED_MODEL_PATH = trained_model_path
        self.BEST_RETRAINED_MODEL_PATH = CURRENT_WORKING_DIR + "/data/weights/new_weights-epoch-" + str(epochs) + ".hdf5"
        
    @classmethod
    def retrain_model(self, training_data, training_labels, test_data, test_labels, test_data_new, test_labels_new, network_config, reverse_data_dict):
        """
        Retrain the trained model with new data and compare performance on test data
        """
        print("New training size: %d" % len(training_labels))
        loaded_model = utils.load_saved_model( NETWORK_C0NFIG_JSON_PATH, self.TRAINED_MODEL_PATH )
        
        print("Old model summary: \n")
        print(loaded_model.summary())

        old_dimensions = loaded_model.layers[0].input_dim
        new_dimensions = training_labels.shape[1]
        
        model = Sequential()
        
        # add embedding
        model.add( Embedding( new_dimensions, network_config[ "embedding_vec_size" ], mask_zero=True ) )     
        model.layers[0].trainable = True
        # initialize embedding layer
        new_embedding_dimensions = model.layers[0].get_weights()[0]
        new_embedding_dimensions[0:old_dimensions,:] = loaded_model.layers[0].get_weights()[0]
        model.layers[0].set_weights([new_embedding_dimensions])
        
        model.add( SpatialDropout1D( network_config[ "dropout" ] ) )

        # add GRU
        model.add( GRU( network_config[ "memory_units" ], dropout=network_config[ "dropout" ], recurrent_dropout=network_config[ "dropout" ], return_sequences=True, activation=network_config[ "activation_recurrent" ] ) )
        # initialize GRU layer
        model.layers[2].set_weights(loaded_model.layers[2].get_weights())
        model.layers[2].trainable = True
        model.add( Dropout( network_config[ "dropout" ] ) )

        # add GRU
        model.add( GRU( network_config[ "memory_units" ], dropout=network_config[ "dropout" ], recurrent_dropout=network_config[ "dropout" ], return_sequences=False, activation=network_config[ "activation_recurrent" ] ) )
        # initialize GRU layer
        model.layers[4].set_weights(loaded_model.layers[4].get_weights())
        model.layers[4].trainable = True	
        model.add( Dropout( network_config[ "dropout" ] ) )

        model.add( Dense(new_dimensions, activation=network_config[ "activation_output" ]))
        model.layers[6].trainable = True
        # initialize output layer
        new_output_dimensions1 = model.layers[6].get_weights()[0]
        new_output_dimensions2 = model.layers[6].get_weights()[1]
        new_output_dimensions1[:, 0:old_dimensions] = loaded_model.layers[6].get_weights()[0]
        new_output_dimensions2[:old_dimensions] = loaded_model.layers[6].get_weights()[1]
        model.layers[6].set_weights([new_output_dimensions1, new_output_dimensions2])
        
        optimizer = RMSprop( lr=network_config[ "learning_rate" ] )
        model.compile( loss=network_config[ "loss_type" ], optimizer=optimizer )
        
        # save the network as json
        utils.save_network( model.to_json(), NETWORK_C0NFIG_JSON_PATH )
        print("New model summary...")
        model.summary()
        
        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint( EPOCH_WEIGHTS_PATH, verbose=0, mode='max' )
        predict_callback_test = PredictCallback( test_data, test_labels, test_data_new, test_labels_new, reverse_data_dict, training_labels.shape[1], test_labels.shape[1], loaded_model, network_config )
        callbacks_list = [ checkpoint, predict_callback_test ]

        reshaped_test_labels = np.zeros([test_labels.shape[0], new_dimensions])
        print("Started training on new data...")
        model_fit_callbacks = model.fit(training_data, training_labels, shuffle="batch", batch_size=network_config[ "batch_size" ], epochs=self.n_epochs, callbacks=callbacks_list)
        print ( "Training finished" )


class PredictCallback( Callback ):
    def __init__( self, x, y, test_data_new, test_labels_new, reverse_data_dictionary, new_dimensions, old_dimensions, loaded_model, network_config ):
        self.test_data = x
        self.test_labels = y
        self.test_data_new = test_data_new
        self.test_labels_new = test_labels_new
        self.reverse_data_dictionary = reverse_data_dictionary
        self.new_dimensions = new_dimensions
        self.old_dimensions = old_dimensions
        self.loaded_model = loaded_model
        self.network_config = network_config

    def on_epoch_end( self, epoch, logs={} ):
        """
        Compute absolute and compatible precision for test data
        """
        #print("Evaluating performance on old test data...")
        #old_precision = utils.verify_model(self.model, self.test_data, self.test_labels, self.reverse_data_dictionary)
        #print("Absolute precision on old test data using new model is: %0.6f" % old_precision)
        
        print("Evaluating performance on test data...")
        new_precision = utils.verify_model(self.model, self.test_data_new, self.test_labels_new, self.reverse_data_dictionary)
        print("Absolute precision on test data using new model is: %0.6f" % new_precision)


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
        "dropout": 0.1,
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
    data_dict = utils.read_file( DATA_DICTIONARY )
    reverse_data_dictionary = utils.read_file( DATA_REV_DICT )

    # get training and test data with their labels
    train_data, train_labels = utils.get_h5_data( TRAIN_DATA )
    test_data, test_labels = utils.get_h5_data( TEST_DATA )
    test_data_new, test_labels_new = utils.get_h5_data( TEST_DATA )

    # execute experiment runs and collect results for each run
    predict_tool.retrain_model( train_data, train_labels, test_data, test_labels, test_data_new, test_labels_new, network_config, reverse_data_dictionary )

    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
