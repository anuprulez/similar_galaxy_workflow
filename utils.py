import os
import numpy as np
import json
import h5py
import datetime
import time

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.core import SpatialDropout1D
from keras.optimizers import RMSprop


def read_file(file_path):
    """
    Read a file
    """
    with open(file_path, "r") as json_file:
        file_content = json.loads(json_file.read())
    return file_content


def write_file(file_path, content):
    """
    Write a file
    """
    remove_file(file_path)
    with open(file_path, "w") as json_file:
        json_file.write(json.dumps(content))

  
def save_processed_workflows(file_path, unique_paths):
    workflow_paths_unique = ""
    for path in unique_paths:
        workflow_paths_unique += path + "\n"
    with open(file_path, "w") as workflows_file:
        workflows_file.write(workflow_paths_unique)


def load_saved_model(model_config, model_weights):
    """
    Load the saved trained model using the saved network and its weights
    """
    # load the network
    loaded_model = model_from_json(model_config)
    # load the saved weights into the model
    loaded_model.set_weights(model_weights)
    return loaded_model


def format_tool_id(tool_link):
    """
    Extract tool id from tool link
    """
    tool_id_split = tool_link.split("/")
    tool_id = tool_id_split[-2] if len(tool_id_split) > 1 else tool_link
    return tool_id


def get_HDF5(hf, d_key):
    """
    Read h5 file to get train and test data
    """
    return hf.get(d_key).value


def save_HDF5(hf_file, d_key, data, d_type=""):
    """
    Save datasets as h5 file
    """
    if (d_type == 'json'): 
        data = json.dumps(data)
    hf_file.create_dataset(d_key, data=data)

      
def set_trained_model(dump_file, model_values):
    """
    Create an h5 file with the trained weights and associated dicts
    """
    hf_file = h5py.File(dump_file, 'w')
    for key in model_values:
        value = model_values[key]
        if key == 'model_weights':
            for idx, item in enumerate(value):
                w_key = "weight_" + str(idx)
                if w_key in hf_file:
                    hf_file.modify(w_key, item)
                else:
                    hf_file.create_dataset(w_key, data=item)
        else:
            if key in hf_file:
                hf_file.modify(key, json.dumps(value))
            else:
                hf_file.create_dataset(key, data=json.dumps(value))
    hf_file.close()


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        
def convert_timestamp(time):
    created_time = datetime.datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
    now_time = datetime.datetime.now()
    month_time = ((now_time - created_time).days) / float(30)
    return month_time


def get_best_parameters(mdl_dict=None):
    """
    Get param values (defaults as well)
    """
    if mdl_dict == None:
        return {
            'lr': 0.001,
            'embedding_size': 256,
            'dropout': 0.1,
            'units': 256,
            'batch_size': 256,
            'loss': "binary_crossentropy",
            'activation_recurrent': "elu",
            'activation_output': "softmax"
        }
    else:
        lr = float(mdl_dict.get("learning_rate", "0.001"))
        embedding_size = int(mdl_dict.get("embedding_vector_size", "256"))
        dropout = float(mdl_dict.get("dropout", "0.1"))
        units = int(mdl_dict.get("memory_units", "256"))
        batch_size = int(mdl_dict.get("batch_size", "256"))
        loss = mdl_dict.get("loss_type", "binary_crossentropy")
        activation_recurrent = mdl_dict.get("activation_recurrent", "elu")
        activation_output = mdl_dict.get("activation_output", "softmax")
    return lr, embedding_size, dropout, units, batch_size, loss, activation_recurrent, activation_output


def set_recurrent_network(mdl_dict, reverse_dictionary):
    """
    Create a RNN network and set its parameters
    """
    dimensions = len(reverse_dictionary) + 1
    lr, embedding_size, dropout, units, batch_size, loss, activation_recurrent, activation_output = get_best_parameters(mdl_dict)
        
    # define the architecture of the recurrent neural network
    model = Sequential()
    model.add(Embedding(dimensions, embedding_size, mask_zero=True))
    model.add(SpatialDropout1D(dropout))
    model.add(GRU(units, dropout=dropout, recurrent_dropout=dropout, return_sequences=True, activation=activation_recurrent))
    model.add(Dropout(dropout))
    model.add(GRU(units, dropout=dropout, recurrent_dropout=dropout, return_sequences=False, activation=activation_recurrent))
    model.add(Dropout(dropout))
    model.add(Dense(dimensions, activation=activation_output))
    optimizer = RMSprop(lr=lr)
    model.compile(loss=loss, optimizer=optimizer)
    return model


def verify_model(model, x, y, reverse_data_dictionary, next_compatible_tools, usage_scores):
    """
    Verify the model on test data
    """
    print("Evaluating performance on test data...")
    print("Test data size: %d" % len(y))
    size = y.shape[0]
    ave_abs_precision = list()
    avg_compatible_pred = list()
    a_tools_usage_scores = list()
    # loop over all the test samples and find prediction precision
    for i in range(size):
        usg_wt_scores = list()
        actual_classes_pos = np.where(y[i] > 0)[0]
        topk = len(actual_classes_pos)
        test_sample = np.reshape(x[i], (1, x.shape[1]))
        test_sample_pos = np.where(x[i] > 0)[0]
        test_sample_tool_pos = x[i][test_sample_pos[0]:]

        # predict next tools for a test path
        prediction = model.predict(test_sample, verbose=0)
        nw_dimension = prediction.shape[1]
        
        # remove the 0th position as there is no tool at this index
        prediction = np.reshape(prediction, (nw_dimension,))

        prediction_pos = np.argsort(prediction, axis=-1)
        topk_prediction_pos = prediction_pos[-topk:]

        # read tool names using reverse dictionary
        sequence_tool_names = [reverse_data_dictionary[int(tool_pos)] for tool_pos in test_sample_tool_pos]
        actual_next_tool_names = [reverse_data_dictionary[int(tool_pos)] for tool_pos in actual_classes_pos]
        top_predicted_next_tool_names = [reverse_data_dictionary[int(tool_pos)] for tool_pos in topk_prediction_pos if int(tool_pos) > 0]
        
        # compute the class weights of predicted tools
        mean_usg_score = 0
        for t_id in topk_prediction_pos:
            t_name = reverse_data_dictionary[int(t_id)]
            if t_id in usage_scores and t_name in actual_next_tool_names:
                usg_wt_scores.append(usage_scores[t_id])
        if len(usg_wt_scores) > 0:
            mean_usg_score = np.mean(usg_wt_scores)
        a_tools_usage_scores.append(mean_usg_score)

        # find false positives
        false_positives = [tool_name for tool_name in top_predicted_next_tool_names if tool_name not in actual_next_tool_names]
        absolute_precision = 1 - (len(false_positives) / float(len(actual_classes_pos)))
        
        # compute precision for tool compatibility
        adjusted_precision = 0.0
        seq_last_tool = sequence_tool_names[-1]
        if seq_last_tool in next_compatible_tools:
            next_tools = next_compatible_tools[seq_last_tool].split(",")
            comp_tools = list(set(false_positives) & set(next_tools))
            for tl in comp_tools:
                adjusted_precision += 1 / float(len(actual_next_tool_names))
            adjusted_precision += absolute_precision
        ave_abs_precision.append(absolute_precision)
        avg_compatible_pred.append(adjusted_precision)
    
    # compute mean across all test samples
    mean_precision = np.mean(ave_abs_precision)
    mean_compatible_precision = np.mean(avg_compatible_pred)
    mean_predicted_usage_score = np.mean(a_tools_usage_scores)
    return mean_precision, mean_compatible_precision, mean_predicted_usage_score
    

def save_model(results, data_dictionary, compatible_next_tools, trained_model_path):
    # save files
    trained_model = results["model"]
    best_model_parameters = results["best_parameters"]
    model_config = trained_model.to_json()
    model_weights = trained_model.get_weights()
    
    model_values = {
        'data_dictionary': data_dictionary,
        'model_config': model_config,
        'best_parameters': best_model_parameters,
        'model_weights': model_weights,
        "compatible_tools": compatible_next_tools
    }
    set_trained_model(trained_model_path, model_values)
