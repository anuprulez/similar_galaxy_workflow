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
import h5py

# machine learning library
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

import extract_workflow_connections
import prepare_data
import optimise_hyperparameters
import utils

warnings.filterwarnings("ignore")


class PredictTool:

    @classmethod
    def __init__( self ):
        """ Init method. """

    @classmethod
    def find_train_best_network(self, network_config, optimise_parameters_node, reverse_dictionary, train_data, train_labels, test_data, test_labels, n_epochs):
        """
        Define recurrent neural network and train sequential data
        """
        # get the best model and train
        print("Start hyperparameter optimisation...")
        hyper_opt = optimise_hyperparameters.HyperparameterOptimisation()
        best_model_parameters = hyper_opt.find_best_model(network_config, optimise_parameters_node, reverse_dictionary, train_data, train_labels, test_data, test_labels)
        print("Best model: %s" % str(best_model_parameters))

        # get the best network
        model = utils.set_recurrent_network(best_model_parameters, reverse_dictionary)
        model.summary()

        # define callbacks
        early_stopping = EarlyStopping(monitor='loss', patience=0, verbose=1, mode='min')
        predict_callback_test = PredictCallback( test_data, test_labels, reverse_dictionary, n_epochs )
        callbacks_list = [ early_stopping, predict_callback_test ]

        print ("Start training on the best model...")
        model_fit_callbacks = model.fit(train_data, train_labels, batch_size=int(best_model_parameters["batch_size"]), epochs=n_epochs, callbacks=callbacks_list, shuffle="batch")
        loss_values = model_fit_callbacks.history[ "loss" ]
        
        return {
            "train_loss": np.array( loss_values ),
            "test_absolute_precision": predict_callback_test.abs_precision,
            "model": model,
            "best_parameters": best_model_parameters
        }
        

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

    if len(sys.argv) != 4:
        print("Usage: python predict_next_tool.py <workflow_file_path> <config_file_path> <trained_model_file_path>")
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
    trained_model_path = sys.argv[3]

    # Extract and process workflows
    connections = extract_workflow_connections.ExtractWorkflowConnections()
    workflow_paths, compatible_next_tools = connections.read_tabular_file(sys.argv[1])

    # Process the paths from workflows
    print ( "Dividing data..." )
    data = prepare_data.PrepareData(int(config["maximum_path_length"]), float(config["test_share"]), retrain)
    train_data, train_labels, test_data, test_labels, data_dictionary, reverse_dictionary = data.get_data_labels_matrices(workflow_paths)

    # execute experiment runs and collect results for each run
    predict_tool = PredictTool()
    results = predict_tool.find_train_best_network(config, optimise_parameters_node, reverse_dictionary, train_data, train_labels, test_data, test_labels, n_epochs)

    # save files
    trained_model = results["model"]
    best_model_parameters = results["best_parameters"]
    model_config = trained_model.to_json()
    model_weights = trained_model.get_weights()
    
    model_values = {
        'compatible_next_tools': compatible_next_tools,
        'data_dictionary': data_dictionary,
        'reverse_dictionary': reverse_dictionary,
        'model_config': model_config,
        'best_parameters': best_model_parameters,
        'model_weights': model_weights
    }
    
    utils.set_trained_model(trained_model_path, model_values)

    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
