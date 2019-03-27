"""
Find the optimal combination of hyperparameters
"""

import numpy as np
import itertools
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.core import SpatialDropout1D
from keras.optimizers import RMSprop

import utils


class HyperparameterOptimisation:

    @classmethod
    def __init__(self):
        """ Init method. """

    @classmethod
    def train_model(self, config, reverse_dictionary, train_data, train_labels, test_data, test_labels, class_weights, train_sample_weights, n_epochs):
        """
        Train a model and report accuracy
        """
        l_recurrent_activations = config["activation_recurrent"].split(",")
        l_output_activations = config["activation_output"].split(",")
        # convert items to integer
        l_batch_size = list(map(int, config["batch_size"].split(",")))
        l_embedding_size = list(map(int, config["embedding_size"].split(",")))
        l_units = list(map(int, config["units"].split(",")))
        # convert items to float
        l_learning_rate = list(map(float, config["learning_rate"].split(",")))
        l_dropout = list(map(float, config["dropout"].split(",")))
        l_spatial_dropout = list(map(float, config["spatial_dropout"].split(",")))
        l_recurrent_dropout = list(map(float, config["recurrent_dropout"].split(",")))
        # get dimensions
        dimensions = len(reverse_dictionary) + 1
        trials = Trials()
        best_model_params = dict()
        
        # specify the search space for finding the best combination of parameters using Bayesian optimisation
        params = {
	    "embedding_size": hp.choice("embedding_size", l_embedding_size),
	    "units": hp.choice("units", l_units),
	    "batch_size": hp.choice("batch_size", l_batch_size),
	    "activation_recurrent": hp.choice("activation_recurrent", l_recurrent_activations),
	    "activation_output": hp.choice("activation_output", l_output_activations),
	    "learning_rate": hp.uniform("learning_rate", l_learning_rate[0], l_learning_rate[1]),
	    "dropout": hp.uniform("dropout", l_dropout[0], l_dropout[1]),
	    "spatial_dropout": hp.uniform("spatial_dropout", l_spatial_dropout[0], l_spatial_dropout[1]),
	    "recurrent_dropout": hp.uniform("recurrent_dropout", l_recurrent_dropout[0], l_recurrent_dropout[1])
        }

        def create_model(params):
            model = Sequential()
            model.add(Embedding(dimensions, params["embedding_size"], mask_zero=True))
            model.add(SpatialDropout1D(params["spatial_dropout"]))
            model.add(GRU(params["units"], dropout=params["dropout"], recurrent_dropout=params["recurrent_dropout"], return_sequences=True, activation=params["activation_recurrent"]))
            model.add(Dropout(params["dropout"]))
            model.add(GRU(params["units"], dropout=params["dropout"], recurrent_dropout=params["recurrent_dropout"], return_sequences=False, activation=params["activation_recurrent"]))
            model.add(Dropout(params["dropout"]))
            model.add(Dense(dimensions, activation=params["activation_output"]))
            optimizer_rms = RMSprop(lr=params["learning_rate"])
            model.compile(loss='binary_crossentropy', optimizer=optimizer_rms)
            
            model.summary()
            model_fit = model.fit(train_data, train_labels,
                batch_size=params["batch_size"],
                epochs=n_epochs,
                shuffle="batch",
                class_weight=class_weights,
                sample_weight=train_sample_weights,
                validation_data=(test_data, test_labels)
            )
            return {'loss': model_fit.history["val_loss"][-1], 'status': STATUS_OK, 'model': model}

        # minimize the objective function using the set of parameters above
        learned_params = fmin(create_model, params, trials=trials, algo=tpe.suggest, max_evals=2)
        
        # set the best params with respective values
        for item in learned_params:
            item_val = learned_params[item]
            if item == 'batch_size':
                best_model_params[item] = l_batch_size[item_val]
            elif item == 'embedding_size':
                best_model_params[item] = l_embedding_size[item_val]
            elif item == 'units':
                best_model_params[item] = l_units[item_val]
            elif item == 'activation_output':
                best_model_params[item] = l_output_activations[item_val]
            elif item == 'activation_recurrent':
                best_model_params[item] = l_recurrent_activations[item_val]
            else:
                best_model_params[item] = item_val
        sorted_results = sorted(trials.results, key=lambda i: i['loss'])
        return sorted_results[0], best_model_params
