"""
Predict next tools in the Galaxy workflows
using machine learning (recurrent neural network)
"""

import sys
import numpy as np
import time
import os
import xml.etree.ElementTree as et
import warnings

# machine learning library
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

import extract_workflow_connections
import prepare_data
import optimise_hyperparameters
import utils

warnings.filterwarnings("ignore")

# file paths
CURRENT_WORKING_DIR = os.getcwd()
NETWORK_C0NFIG_JSON_PATH = CURRENT_WORKING_DIR + "/data/generated_files/model.json"
#EPOCH_WEIGHTS_PATH = CURRENT_WORKING_DIR + "/data/generated_files/weights-epoch-{epoch:d}.hdf5"
EPOCH_WEIGHTS_PATH = CURRENT_WORKING_DIR + "/data/generated_files/trained_model.hdf5"
DATA_REV_DICT = CURRENT_WORKING_DIR + "/data/generated_files/data_rev_dict.txt"
DATA_DICTIONARY = CURRENT_WORKING_DIR + "/data/generated_files/data_dictionary.txt"
BEST_PARAMETERS = CURRENT_WORKING_DIR + "/data/generated_files/best_params.json"
MEAN_TEST_ABSOLUTE_PRECISION = CURRENT_WORKING_DIR + "/data/generated_files/mean_test_absolute_precision.txt"
MEAN_TRAIN_LOSS = CURRENT_WORKING_DIR + "/data/generated_files/mean_test_loss.txt"
WORKFLOW_PATHS_FILE = CURRENT_WORKING_DIR + "/data/generated_files/workflow_connections_paths.txt"
COMPATIBLE_NEXT_TOOLS = CURRENT_WORKING_DIR + "/data/generated_files/compatible_tools.json"
TRAIN_DATA_CLASS_FREQ = CURRENT_WORKING_DIR + "/data/generated_files/train_data_class_freq.txt"


class PredictTool:

    @classmethod
    def __init__( self, n_epochs ):
        """ Init method. """
        self.BEST_MODEL_PATH = CURRENT_WORKING_DIR + "/data/weights/weights-epoch-" + str(n_epochs) + ".hdf5"

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
        early_stopping = EarlyStopping(monitor='loss', patience=0, verbose=1, mode='min')
        checkpoint = ModelCheckpoint(EPOCH_WEIGHTS_PATH, verbose=0, mode='max')
        predict_callback_test = PredictCallback( test_data, test_labels, reverse_dictionary, n_epochs )
        callbacks_list = [ checkpoint, early_stopping, predict_callback_test ]

        print ("Start training on the best model...")
        model_fit_callbacks = model.fit(train_data, train_labels, batch_size=int(best_model_parameters["batch_size"]), epochs=n_epochs, callbacks=callbacks_list, shuffle="batch")
        loss_values = model_fit_callbacks.history[ "loss" ]
        
        return {
            "train_loss": np.array( loss_values ),
            "test_absolute_precision": predict_callback_test.abs_precision,
        }
        print ("Training finished")
        

class PredictCallback( Callback ):
    def __init__( self, test_data, test_labels, reverse_data_dictionary, n_epochs ):
        self.test_data = test_data
        self.test_labels = test_labels
        self.reverse_data_dictionary = reverse_data_dictionary
        self.abs_precision = list()

    def on_epoch_end( self, epoch, logs={} ):
        """
        Compute absolute and compatible precision for test data
        """
        mean_precision = utils.verify_model(self.model, self.test_data, self.test_labels, self.reverse_data_dictionary)
        self.abs_precision.append(mean_precision)
        print( "Epoch %d topk absolute precision: %.2f" % ( epoch + 1, mean_precision ) )


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
    connections = extract_workflow_connections.ExtractWorkflowConnections()
    workflow_paths, compatible_next_tools = connections.read_tabular_file(sys.argv[1])

    # Process the paths from workflows
    print ( "Dividing data..." )
    data = prepare_data.PrepareData(int(config["maximum_path_length"]), float(config["test_share"]), retrain)
    train_data, train_labels, test_data, test_labels, data_dictionary, reverse_dictionary = data.get_data_labels_matrices(workflow_paths)

    # execute experiment runs and collect results for each run
    predict_tool = PredictTool(n_epochs)
    results = predict_tool.find_train_best_network(config, optimise_parameters_node, reverse_dictionary, train_data, train_labels, test_data, test_labels)

    np.savetxt( MEAN_TEST_ABSOLUTE_PRECISION, results[ "test_absolute_precision" ], delimiter="," )
    np.savetxt( MEAN_TRAIN_LOSS, results[ "train_loss" ], delimiter="," )

    # save files
    utils.write_file(COMPATIBLE_NEXT_TOOLS, compatible_next_tools)
    utils.save_processed_workflows(WORKFLOW_PATHS_FILE, workflow_paths)
    utils.write_file(DATA_DICTIONARY, data_dictionary)
    utils.write_file(DATA_REV_DICT, reverse_dictionary)
    # utils.write_file(TRAIN_DATA_CLASS_FREQ, frequency_scores)

    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
