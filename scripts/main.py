"""
Predict next tools in the Galaxy workflows
using machine learning (recurrent neural network)
"""

import numpy as np
import time
import xml.etree.ElementTree as et
import warnings
import argparse

# machine learning library
import keras.callbacks as callbacks
from tensorflow_model_optimization.sparsity import keras as sparsity

import extract_workflow_connections
import prepare_data
import optimise_hyperparameters
import utils

warnings.filterwarnings("ignore")


class PredictTool:

    @classmethod
    def __init__(self):
        """ Init method. """

    @classmethod
    def find_train_best_network(self, network_config, reverse_dictionary, train_data, train_labels, test_data, test_labels, n_epochs, class_weights, usage_pred, compatible_next_tools):
        """
        Define recurrent neural network and train sequential data
        """
        '''print("Start hyperparameter optimisation...")
        hyper_opt = optimise_hyperparameters.HyperparameterOptimisation()
        best_params = hyper_opt.train_model(network_config, reverse_dictionary, train_data, train_labels, class_weights)'''
        best_params = dict()
        # retrieve the model and train on complete dataset without validation set
        model, best_params = utils.set_recurrent_network(best_params, reverse_dictionary, class_weights)

        # define callbacks
        predict_callback_test = PredictCallback(test_data, test_labels, reverse_dictionary, n_epochs, compatible_next_tools, usage_pred)

        callbacks_list = [predict_callback_test, sparsity.UpdatePruningStep(), sparsity.PruningSummaries(log_dir='data')]

        print("Start training on the best model...")
        model_fit = model.fit(
            train_data,
            train_labels,
            batch_size=int(best_params["batch_size"]),
            epochs=n_epochs,
            verbose=1,
            callbacks=callbacks_list,
            shuffle="batch",
            validation_data=(test_data, test_labels)
        )

        train_performance = {
            "train_loss": np.array(model_fit.history["loss"]),
            "model": model,
            "best_parameters": best_params
        }

        # if there is test data, add more information
        if len(test_data) > 0:
            train_performance["validation_loss"] = np.array(model_fit.history["val_loss"])
            train_performance["precision"] = predict_callback_test.precision
            train_performance["usage_weights"] = predict_callback_test.usage_weights
        return train_performance


class PredictCallback(callbacks.Callback):
    def __init__(self, test_data, test_labels, reverse_data_dictionary, n_epochs, next_compatible_tools, usg_scores):
        self.test_data = test_data
        self.test_labels = test_labels
        self.reverse_data_dictionary = reverse_data_dictionary
        self.precision = list()
        self.usage_weights = list()
        self.n_epochs = n_epochs
        self.next_compatible_tools = next_compatible_tools
        self.pred_usage_scores = usg_scores

    def on_epoch_end(self, epoch, logs={}):
        """
        Compute absolute and compatible precision for test data
        """
        if len(self.test_data) > 0:
            precision, usage_weights = utils.verify_model(self.model, self.test_data, self.test_labels, self.reverse_data_dictionary, self.next_compatible_tools, self.pred_usage_scores)
            self.precision.append(precision)
            self.usage_weights.append(usage_weights)
            print("Epoch %d precision: %s" % (epoch + 1, precision))
            print("Epoch %d usage weights: %s" % (epoch + 1, usage_weights))


if __name__ == "__main__":
    start_time = time.time()

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-wf", "--workflow_file", required=True, help="workflows tabular file")
    arg_parser.add_argument("-cf", "--config_file", required=True, help="configuration file")
    arg_parser.add_argument("-tm", "--trained_model_file", required=True, help="trained model file")
    arg_parser.add_argument("-tu", "--tool_usage_file", required=True, help="tool usage file")
    arg_parser.add_argument("-cd", "--cutoff_date", required=True, help="earliest date for taking tool usage")
    arg_parser.add_argument("-pl", "--maximum_path_length", required=True, help="maximum length of tool path")
    args = vars(arg_parser.parse_args())

    # get argument values
    maximum_path_length = int(args["maximum_path_length"])
    trained_model_path = args["trained_model_file"]
    tool_usage_path = args["tool_usage_file"]
    cutoff_date = args["cutoff_date"]

    # read config parameters
    tree = et.parse(args["config_file"])
    root = tree.getroot()
    config = dict()
    optimise_parameters_node = None
    for child in root:
        if child.tag == "optimise_parameters":
            optimise_parameters_node = child
        for item in child:
            config[item.get("name")] = item.get("value")
    n_epochs = int(config["n_epochs"])
    test_share = float(config["test_share"])

    # Extract and process workflows
    connections = extract_workflow_connections.ExtractWorkflowConnections()
    workflow_paths, compatible_next_tools = connections.read_tabular_file(args["workflow_file"])

    # Process the paths from workflows
    print("Dividing data...")
    data = prepare_data.PrepareData(maximum_path_length, test_share)
    train_data, train_labels, test_data, test_labels, data_dictionary, reverse_dictionary, class_weights, usage_pred = data.get_data_labels_matrices(workflow_paths, tool_usage_path, cutoff_date, compatible_next_tools)

    # find the best model and start training
    predict_tool = PredictTool()

    # start training with weighted classes
    print("Training with weighted classes and samples ...")
    results_weighted = predict_tool.find_train_best_network(config, reverse_dictionary, train_data, train_labels, test_data, test_labels, n_epochs, class_weights, usage_pred, compatible_next_tools)
    utils.save_model(results_weighted, data_dictionary, compatible_next_tools, trained_model_path, class_weights)
    end_time = time.time()
    print("Program finished in %s seconds" % str(end_time - start_time))
