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
EPOCH_WEIGHTS_PATH = CURRENT_WORKING_DIR + "/data/weights/weights-epoch-{epoch:d}.hdf5"
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
    def retrain_model(self, training_data, training_labels, test_data, test_labels, network_config, reverse_data_dict):
        """
        Retrain the trained model with new data and compare performance on test data
        """
        print("New training size: %d" % len(training_labels))
        loaded_model = utils.load_saved_model( NETWORK_C0NFIG_JSON_PATH, self.TRAINED_MODEL_PATH )

        layer_names = [layer.name for layer in loaded_model.layers]
 
        print("Old model summary: \n")
        print(loaded_model.summary())

        old_dimensions = loaded_model.layers[0].input_dim
        new_dimensions = training_labels.shape[1]

        # model configurations
        dropout = network_config["dropout"]
        memory_units = network_config[ "memory_units" ]
        act_recurrent = network_config["activation_recurrent"]
        embedding_size = network_config["embedding_vec_size"]
        act_output = network_config["activation_output"]
        learning_rate = network_config["learning_rate"]
        loss_type = network_config["loss_type"]

        model = Sequential()
        
        for idx, ly in enumerate(layer_names):
            if "embedding" in ly:
                model.add( Embedding(new_dimensions, embedding_size, mask_zero=True))     
                model_layer = model.layers[idx]
                model_layer.trainable = True
                # initialize embedding layer
                new_embedding_dimensions = model_layer.get_weights()[0]
                new_embedding_dimensions[0:old_dimensions,:] = loaded_model.layers[idx].get_weights()[0]
                model_layer.set_weights([new_embedding_dimensions])
            elif "spatial_dropout1d_1" in ly:
                model.add( SpatialDropout1D(dropout))
            elif "dropout" in ly:
                model.add( Dropout(dropout))
            elif "gru" in ly:
                return_seq = True if "gru_1" in ly else False
                model.add( GRU(memory_units, dropout=dropout, recurrent_dropout=dropout, return_sequences=return_seq, activation=act_recurrent))
                model_layer = model.layers[idx]
                # initialize GRU layer
                model_layer.set_weights(loaded_model.layers[idx].get_weights())
                model_layer.trainable = True
            elif "dense" in ly:
                model.add( Dense(new_dimensions, activation=act_output))
                model_layer = model.layers[idx]
                model_layer.trainable = True
                # initialize output layer
                new_output_dimensions1 = model_layer.get_weights()[0]
                new_output_dimensions2 = model_layer.get_weights()[1]
                new_output_dimensions1[:, 0:old_dimensions] = loaded_model.layers[6].get_weights()[0]
                new_output_dimensions2[:old_dimensions] = loaded_model.layers[6].get_weights()[1]
                model_layer.set_weights([new_output_dimensions1, new_output_dimensions2])
                
        optimizer = RMSprop(lr=learning_rate)
        model.compile(loss=loss_type, optimizer=optimizer)
        
        # save the network as json
        utils.save_network( model.to_json(), NETWORK_C0NFIG_JSON_PATH )
        print("New model summary...")
        model.summary()
        
        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint( EPOCH_WEIGHTS_PATH, verbose=0, mode='max' )
        predict_callback_test = PredictCallback( test_data, test_labels, reverse_data_dict, loaded_model )
        callbacks_list = [ checkpoint, predict_callback_test ]

        reshaped_test_labels = np.zeros([test_labels.shape[0], new_dimensions])
        print("Started training on new data...")
        model_fit_callbacks = model.fit(training_data, training_labels, shuffle="batch", batch_size=network_config[ "batch_size" ], epochs=self.n_epochs, callbacks=callbacks_list)
        print ( "Training finished" )


class PredictCallback( Callback ):
    def __init__( self, test_data, test_labels, reverse_data_dictionary, loaded_model ):
        self.test_data = test_data
        self.test_labels = test_labels
        self.reverse_data_dictionary = reverse_data_dictionary
        self.loaded_model = loaded_model

    def on_epoch_end( self, epoch, logs={} ):
        """
        Compute absolute and compatible precision for test data
        """ 
        print("Evaluating performance on test data...")
        new_precision = utils.verify_model(self.model, self.test_data, self.test_labels, self.reverse_data_dictionary)
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

    # execute experiment runs and collect results for each run
    predict_tool.retrain_model( train_data, train_labels, test_data, test_labels, network_config, reverse_data_dictionary )

    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
