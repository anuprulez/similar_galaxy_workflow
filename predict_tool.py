"""
Predict next tools in the Galaxy workflows
using machine learning (recurrent neural network)
"""

import sys
import numpy as np
import time
import xml.etree.ElementTree as et
import warnings

# machine learning library
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import log_loss

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
    def find_train_best_network(self, network_config, optimise_parameters_node, reverse_dictionary, train_data, train_labels, test_data, test_labels, class_weights, usage_pred, compatible_next_tools):
        """
        Define recurrent neural network and train sequential data
        """
        # get the best model and train
        print("Start hyperparameter optimisation...")
        hyper_opt = optimise_hyperparameters.HyperparameterOptimisation()
        best_params = hyper_opt.train_model(network_config, reverse_dictionary, train_data, train_labels, test_data, test_labels, class_weights)
        
        utils.write_file("data/generated_files/best_params.txt", best_params)
        
        classifier = RandomForestClassifier(
            n_estimators=int(best_params["n_estimators"]),
            max_depth=int(best_params["max_depth"]),
            min_samples_split=float(best_params["min_samples_split"]),
            min_samples_leaf=float(best_params["min_samples_leaf"]),
            class_weight=[{0: w} for w in list(class_weights.values())]
        )
        classifier.fit(train_data, train_labels)

        if len(test_data) > 0:
            precision, usage_weights = utils.verify_model(classifier, test_data, test_labels, reverse_dictionary, compatible_next_tools, usage_pred)
            print("Precision: %s" % precision)
            print("Usage weights: %s" % usage_weights)


if __name__ == "__main__":

    if len(sys.argv) != 6:
        print("Usage: python predict_next_tool.py <workflow_file_path> <config_file_path> <trained_model_file_path> <tool_usage_data> '<cutoff date as yyyy-mm-dd>'")
        exit(1)
    start_time = time.time()

    # read config parameters
    tree = et.parse(sys.argv[2])
    root = tree.getroot()
    config = dict()
    optimise_parameters_node = None
    for child in root:
        if child.tag == "optimise_parameters":
            optimise_parameters_node = child
        for item in child:
            config[item.get("name")] = item.get("value")
    maximum_path_length = 25
    test_share = float(config["test_share"])
    trained_model_path = sys.argv[3]
    tool_usage_path = sys.argv[4]
    cutoff_date = sys.argv[5]

    # Extract and process workflows
    connections = extract_workflow_connections.ExtractWorkflowConnections()
    workflow_paths, compatible_next_tools, frequency_paths = connections.read_tabular_file(sys.argv[1])

    # Process the paths from workflows
    print("Dividing data...")
    data = prepare_data.PrepareData(maximum_path_length, test_share)
    train_data, train_labels, test_data, test_labels, data_dictionary, reverse_dictionary, class_weights, train_sample_weights, usage_pred = data.get_data_labels_matrices(workflow_paths, frequency_paths, tool_usage_path, cutoff_date)

    # find the best model and start training
    predict_tool = PredictTool()

    # start training with weighted classes
    print("Training with weighted classes and samples ...")
    results_weighted = predict_tool.find_train_best_network(config, optimise_parameters_node, reverse_dictionary, train_data, train_labels, test_data, test_labels, class_weights, usage_pred, compatible_next_tools)

    end_time = time.time()
    print("Program finished in %s seconds" % str(end_time - start_time))
