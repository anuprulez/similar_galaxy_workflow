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
from keras.callbacks import Callback
from keras.layers.core import SpatialDropout1D
from keras.optimizers import RMSprop
import xml.etree.ElementTree as et

import extract_workflow_connections
import prepare_data
import utils


# file paths
CURRENT_WORKING_DIR = os.getcwd()
NETWORK_C0NFIG_JSON_PATH = CURRENT_WORKING_DIR + "/data/generated_files/model.json"
EPOCH_WEIGHTS_PATH = CURRENT_WORKING_DIR + "/data/generated_files/trained_model.hdf5"
DATA_REV_DICT = CURRENT_WORKING_DIR + "/data/generated_files/data_rev_dict.txt"
DATA_DICTIONARY = CURRENT_WORKING_DIR + "/data/generated_files/data_dictionary.txt"
BEST_PARAMETERS = CURRENT_WORKING_DIR + "/data/generated_files/best_params.json"
MEAN_TEST_ABSOLUTE_PRECISION = CURRENT_WORKING_DIR + "/data/generated_files/retrain_mean_test_absolute_precision.txt"
MEAN_TRAIN_LOSS = CURRENT_WORKING_DIR + "/data/generated_files/retrain_mean_test_loss.txt"
TRAIN_DUMP_FILE = CURRENT_WORKING_DIR + "/data/generated_files/train_dump.hdf5"


class RetrainPredictTool:

    @classmethod
    def __init__( self ):
        """ Init method. """
        
    @classmethod
    def retrain_model(self, training_data, training_labels, test_data, test_labels, reverse_data_dict, epochs, loaded_model, best_params):
        """
        Retrain the trained model with new data and compare performance on test data
        """
        print("New training size: %d" % len(training_labels))
        #loaded_model = utils.load_saved_model(NETWORK_C0NFIG_JSON_PATH, self.TRAINED_MODEL_PATH)

        layer_names = [layer.name for layer in loaded_model.layers]

        print("Old model summary: \n")
        print(loaded_model.summary())

        old_dimensions = loaded_model.layers[0].input_dim
        new_dimensions = training_labels.shape[1]

        # best model configurations
        lr, embedding_size, dropout, units, batch_size, loss, act_recurrent, act_output = utils.get_defaults(best_params)

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
                model.add(SpatialDropout1D(dropout))
            elif "dropout" in ly:
                model.add(Dropout(dropout))
            elif "gru" in ly:
                layer = loaded_model.layers[idx]
                model.add(GRU(units, dropout=dropout, recurrent_dropout=dropout, return_sequences=layer.return_sequences, activation=act_recurrent))
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

        model.compile(loss=loss, optimizer=RMSprop(lr=lr))
        
        # save the network as json
        print("New model summary...")
        model.summary()
        
        # create checkpoint after each epoch - save the weights to h5 file
        # checkpoint = ModelCheckpoint( EPOCH_WEIGHTS_PATH, verbose=0, mode='max' )
        predict_callback_test = PredictCallback( test_data, test_labels, reverse_data_dict, epochs )
        callbacks_list = [ predict_callback_test ]
        
        print("Started training on new data...")
        model_fit_callbacks = model.fit(training_data, training_labels, shuffle="batch", batch_size=int(best_params["batch_size"]), epochs=epochs, callbacks=callbacks_list)
        loss_values = model_fit_callbacks.history[ "loss" ]
        
        print ( "Training finished" )
        
        return {
            "train_loss": np.array( loss_values ),
            "test_absolute_precision": predict_callback_test.abs_precision,
            "model": model,
            "best_params": best_params
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
        print( "Usage: python retrain_model.py <workflow_file_path> <training_epochs> <trained_model_path>" )
        exit( 1 )
    start_time = time.time()

    tree = et.parse(sys.argv[3])
    root = tree.getroot()
    config = dict()
    for child in root:
        if child.tag == "ml_parameters":
            for item in child:
                config[item.get("name")] = item.get("value")

    n_epochs_retrain = int(config['n_epochs_retrain'])
    retrain = True
    
    # Extract the previous model
    
    hf_file = h5py.File(TRAIN_DUMP_FILE, 'r')
    model_config = json.loads(utils.get_HDF5(hf_file, 'model_config'))
    old_data_dictionary = json.loads(utils.get_HDF5(hf_file, 'data_dictionary'))
    best_parameters = json.loads(utils.get_HDF5(hf_file, 'best_parameters'))
    model_weights = list()
    weight_ctr = 0
    while True:
        try:
            d_key = "weight_" + str(weight_ctr)
            weights = utils.get_h5_data(hf_file, d_key)
            model_weights.append(weights)
            weight_ctr += 1
        except:
            break
    hf_file.close()
    
    loaded_model = utils.load_saved_model(model_config, model_weights)
    
    # Extract and process workflows
    connections = extract_workflow_connections.ExtractWorkflowConnections()
    workflow_paths, compatible_next_tools = connections.read_tabular_file(sys.argv[1])

    # Process the paths from workflows
    print ("Dividing data...")
    data = prepare_data.PrepareData(int(config["maximum_path_length"]), float(config["test_share"]), retrain)
    train_data, train_labels, test_data, test_labels, data_dictionary, reverse_dictionary = data.get_data_labels_matrices(workflow_paths, old_data_dictionary)

    # retrain the model on new data
    retrain_predict_tool = RetrainPredictTool()
    results = retrain_predict_tool.retrain_model(train_data, train_labels, test_data, test_labels, reverse_dictionary, n_epochs_retrain, loaded_model, best_parameters)
    
    # save the latest model
    trained_model = results["model"]
    best_model_parameters = results["best_params"]
    model_values = {
        'compatible_next_tools': compatible_next_tools,
        'data_dictionary': data_dictionary,
        'reverse_dictionary': reverse_dictionary,
        'model_config': trained_model.to_json(),
        'best_parameters': best_model_parameters,
        'model_weights': trained_model.get_weights()
    }
    
    utils.set_trained_model(TRAIN_DUMP_FILE, model_values)
    
    #np.savetxt( MEAN_TEST_ABSOLUTE_PRECISION, results[ "test_absolute_precision" ], delimiter="," )
    #np.savetxt( MEAN_TRAIN_LOSS, results[ "train_loss" ], delimiter="," )

    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ))
