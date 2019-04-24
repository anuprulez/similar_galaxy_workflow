import os
import numpy as np
import json
import h5py

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


def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def extract_configuration(config_object):
    config_loss = dict()
    for index, item in enumerate(config_object):
        config_loss[index] = list()
        d_config = dict()
        d_config['loss'] = item['result']['loss']
        d_config['params_config'] = item['misc']['vals']
        config_loss[index].append(d_config)
    return config_loss
 
def compute_precision(model, x, y, reverse_data_dictionary, next_compatible_tools, usage_scores, actual_classes_pos, topk, is_absolute=False):
    """
    Compute absolute and compatible precision
    """
    absolute_precision = 0.0
    compatible_precision = 0.0
    test_sample = np.reshape(x, (1, len(x)))
    test_sample_pos = np.where(x > 0)[0]
    test_sample_tool_pos = x[test_sample_pos[0]:]

    # predict next tools for a test path
    prediction = model.predict(test_sample)
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
    usg_wt_scores = list()
    for t_id in topk_prediction_pos:
        t_name = reverse_data_dictionary[int(t_id)]
        if t_id in usage_scores and t_name in actual_next_tool_names:
            usg_wt_scores.append(usage_scores[t_id])
    if len(usg_wt_scores) > 0:
            mean_usg_score = np.mean(usg_wt_scores)
    # absolute or compatible precision 
    if is_absolute == True:
        false_positives = [tool_name for tool_name in top_predicted_next_tool_names if tool_name not in actual_next_tool_names]
        absolute_precision = 1 - (len(false_positives) / float(topk))
    else:
        seq_last_tool = sequence_tool_names[-1]
        if seq_last_tool in next_compatible_tools:
            next_tools = next_compatible_tools[seq_last_tool].split(",")
            comp_tools = list(set(top_predicted_next_tool_names) & set(next_tools))
            compatible_precision = len(comp_tools) / float(len(top_predicted_next_tool_names))
        else:
            compatible_precision = 0.0
    return mean_usg_score, absolute_precision, compatible_precision


def verify_model(model, x, y, reverse_data_dictionary, next_compatible_tools, usage_scores, topk_list=[1, 2, 3]):
    """
    Verify the model on test data
    """
    print("Evaluating performance on test data...")
    print("Test data size: %d" % len(y))
    size = y.shape[0]
    precision = np.zeros([len(y), len(topk_list) + 1])
    usage_weights = np.zeros([len(y), len(topk_list) + 1])
    # loop over all the test samples and find prediction precision
    for i in range(size):
        actual_classes_pos = np.where(y[i] > 0)[0]
        abs_topk = len(actual_classes_pos)
        abs_mean_usg_score, absolute_precision, _ = compute_precision(model, x[i,:], y, reverse_data_dictionary, next_compatible_tools, usage_scores, actual_classes_pos, abs_topk, True)
        precision[i][0] = absolute_precision
        usage_weights[i][0] = abs_mean_usg_score
        for index, comp_topk in enumerate(topk_list):
            compatible_mean_usg_score, _, compatible_precision = compute_precision(model, x[i,:], y, reverse_data_dictionary, next_compatible_tools, usage_scores, actual_classes_pos, comp_topk)
            precision[i][index+1] = compatible_precision
            usage_weights[i][index+1] = compatible_mean_usg_score
    mean_precision = np.mean(precision, axis=0)
    mean_usage = np.mean(usage_weights, axis=0)
    return mean_precision, mean_usage
