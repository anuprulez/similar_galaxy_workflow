"""
Find the optimal combination of hyperparameters
"""

import numpy as np
import itertools
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

import utils


class HyperparameterOptimisation:

    @classmethod
    def __init__(self):
        """ Init method. """

    @classmethod
    def train_model(self, config, reverse_dictionary, train_data, train_labels, test_data, test_labels, class_weights):
        """
        Train a model and report accuracy
        """        
        l_n_estimators = list(map(int, config["n_estimators"].split(",")))
        l_max_depth = list(map(int, config["max_depth"].split(",")))
        l_min_samples_split = list(map(float, config["min_samples_split"].split(",")))
        l_min_samples_leaf = list(map(float, config["min_samples_leaf"].split(",")))

        validation_split = float(config["validation_split"])

        # specify the search space for finding the best combination of parameters using Bayesian optimisation
        params = {	    
	    "n_estimators": hp.quniform("n_estimators", l_n_estimators[0], l_n_estimators[1], 1),
	    "max_depth": hp.quniform("max_depth", l_max_depth[0], l_max_depth[1], 1),
	    "min_samples_split": hp.uniform("min_samples_split", l_min_samples_split[0], l_min_samples_split[1]),
	    "min_samples_leaf": hp.uniform("min_samples_leaf", l_min_samples_leaf[0], l_min_samples_leaf[1]),
        }

        def create_model(params):
            clf = RandomForestClassifier(
                n_estimators=int(params["n_estimators"]),
                max_depth=int(params["max_depth"]),
                class_weight=[{0: w} for w in list(class_weights.values())]
            )
            clf.fit(train_data, train_labels)
            test_pred_labels = clf.predict(test_data)
            loss = log_loss(test_labels, test_pred_labels)
            return {'loss': loss, 'status': STATUS_OK}

        # minimize the objective function using the set of parameters above
        trials = Trials()
        learned_params = fmin(create_model, params, trials=trials, algo=tpe.suggest, max_evals=int(config["max_evals"]))
        model_config = utils.extract_configuration(trials.trials)
        utils.write_file("data/generated_files/trials.txt", model_config)
        print(learned_params)
        return learned_params
