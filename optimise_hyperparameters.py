"""
Find the optimal combination of hyperparameters
"""
from keras.callbacks import EarlyStopping

import numpy as np
import itertools

import utils


class HyperparameterOptimisation:

    @classmethod
    def __init__(self):
        """ Init method. """

    @classmethod
    def make_combinations(self, optimize_parameters):
        """
        Make all possible combinations
        """
        parameters = dict()
        parameter_names = list()
        parameter_values = list()
        for item in optimize_parameters:
            values = item.get("value")
            parameters[item.get("name")] = values.split(",")
        possible_values = 1
        for pt in parameters:
             parameter_names.append(pt)
             parameter_values.append(parameters[pt])
             possible_values *= len(parameters[pt])
        models = list(itertools.product(*parameter_values))
        return parameter_names, models
        
    @classmethod
    def train_model(self, mdl_dict, n_epochs_optimise, reverse_dictionary, train_data, train_labels, test_data, test_labels):
        """
        Train a model and report accuracy
        """
        # get the network
        model = utils.set_recurrent_network(mdl_dict, reverse_dictionary)
        
        model.summary()
        
        early_stopping = EarlyStopping(monitor='loss', patience=0, verbose=1, mode='min')
        
        # fit the model
        model_fit_callbacks = model.fit(train_data, train_labels, batch_size=int(mdl_dict["batch_size"]), epochs=n_epochs_optimise, shuffle="batch", callbacks=[early_stopping])

        # verify model with test data
        mean_precision = utils.verify_model(model, test_data, test_labels, reverse_dictionary)
        return mean_precision
        
    @classmethod
    def find_best_model(self, network_config, optimise_parameters_node, reverse_dictionary, train_data, train_labels, test_data, test_labels):
        """
        Collect the accuracies of all model and find the best one
        """
        parameter_names, models = self.make_combinations(optimise_parameters_node)
        n_epochs_optimise = int(network_config["n_epochs_optimise"])
        n_models = len(models)
        model_performances = list()
        for mdl_idx, mdl in enumerate(models):
            mdl_dict = dict(zip(parameter_names, list(mdl)))
            print("Training on model(%d/%d): \n%s" % (mdl_idx + 1, n_models, str(mdl_dict)))
            model_accuracy = self.train_model(mdl_dict, n_epochs_optimise, reverse_dictionary, train_data, train_labels, test_data, test_labels)
            model_performances.append(model_accuracy)
        best_model_idx = np.argsort(model_performances)[-1]
        return dict(zip(parameter_names, list(models[best_model_idx])))
