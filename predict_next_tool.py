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
import xml.etree.ElementTree as et
import warnings

# machine learning library
from keras.callbacks import ModelCheckpoint

import extract_workflow_connections
import prepare_data
import optimise_hyperparameters
import utils

warnings.filterwarnings("ignore")

# file paths
CURRENT_WORKING_DIR = os.getcwd()
NETWORK_C0NFIG_JSON_PATH = CURRENT_WORKING_DIR + "/data/generated_files/model.json"
EPOCH_WEIGHTS_PATH = CURRENT_WORKING_DIR + "/data/weights/weights-epoch-{epoch:d}.hdf5"
DATA_REV_DICT = CURRENT_WORKING_DIR + "/data/generated_files/data_rev_dict.txt"
DATA_DICTIONARY = CURRENT_WORKING_DIR + "/data/generated_files/data_dictionary.txt"
TRAIN_DATA = CURRENT_WORKING_DIR + "/data/generated_files/train_data.h5"
TEST_DATA = CURRENT_WORKING_DIR + "/data/generated_files/test_data.h5"
BEST_PARAMETERS = CURRENT_WORKING_DIR + "/data/generated_files/best_params.json"


class PredictNextTool:

    @classmethod
    def __init__( self, n_epochs ):
        """ Init method. """
        self.BEST_MODEL_PATH = CURRENT_WORKING_DIR + "/data/weights/weights-epoch-"+ str(n_epochs) + ".hdf5"

    @classmethod
    def find_train_best_network(self, network_config, optimise_parameters_node, reverse_dictionary, train_data, train_labels, test_data, test_labels):
        """
        Define recurrent neural network and train sequential data
        """
        # get the best model and train
        print("Start hyperparameter optimisation...")
        hyper_opt = optimise_hyperparameters.HyperparameterOptimisation()
        best_model_parameters = hyper_opt.find_best_model(network_config, optimise_parameters_node, reverse_dictionary, train_data, train_labels, test_data, test_labels)
        print("Best model: %s" % str(best_model_parameters))

        utils.write_file( BEST_PARAMETERS, best_model_parameters )
        
        # get the best network
        model = utils.set_recurrent_network(best_model_parameters, reverse_dictionary)
    
        # save the network configuration
        utils.save_network(model.to_json(), NETWORK_C0NFIG_JSON_PATH)
        model.summary()

        # create checkpoint after each epoch - save the weights to h5 file
        checkpoint = ModelCheckpoint(EPOCH_WEIGHTS_PATH, verbose=0, mode='max')
        callbacks_list = [checkpoint]
        print ("Start training on the best model...")
        model.fit(train_data, train_labels, batch_size=int(best_model_parameters["batch_size"]), epochs=n_epochs, callbacks=callbacks_list, shuffle="batch")
        print ("Training finished")


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python predict_next_tool.py <workflow_file_path> <config_file_path>")
        exit( 1 )
    start_time = time.time()

    # read config parameters
    tree = et.parse( sys.argv[2] )
    root = tree.getroot()
    config = dict()
    optimise_parameters_node = None
    for child in root:
        if child.tag == "optimise_parameters":
            optimise_parameters_node = child
        for item in child:
            config[item.get("name")] = item.get("value")

    n_epochs = int(config['n_epochs'])
    retrain = config['retrain']
    
    # Extract and process workflows
    connections = extract_workflow_connections.ExtractWorkflowConnections(sys.argv[1], retrain)
    connections.read_tabular_file()

    # Process the paths from workflows
    print ( "Dividing data..." )
    data = prepare_data.PrepareData(int(config["maximum_path_length"]), float(config["test_share"]), retrain)
    data.get_data_labels_mat()
    
    # get data dictionary
    reverse_data_dictionary = utils.read_file( DATA_REV_DICT)
    
    # get training and test data with their labels
    train_data, train_labels = utils.get_h5_data( TRAIN_DATA )
    test_data, test_labels = utils.get_h5_data( TEST_DATA )

    # execute experiment runs and collect results for each run
    predict_tool = PredictNextTool(n_epochs)
    predict_tool.find_train_best_network(config, optimise_parameters_node, reverse_data_dictionary, train_data, train_labels, test_data, test_labels)
    loaded_model = utils.load_saved_model(NETWORK_C0NFIG_JSON_PATH, predict_tool.BEST_MODEL_PATH)
    
    # verify the model with test data
    mean_precision = utils.verify_model(loaded_model, test_data, test_labels, reverse_data_dictionary)

    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
