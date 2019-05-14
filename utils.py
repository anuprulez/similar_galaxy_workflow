import os
import numpy as np
import json
import h5py

from keras.models import model_from_json, Sequential
from keras.layers import Dense, Dropout, Conv1D, GlobalMaxPooling1D, MaxPooling1D, Flatten, SpatialDropout1D
from keras.layers.embeddings import Embedding
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


def extract_configuration(config_object):
    config_loss = dict()
    for index, item in enumerate(config_object):
        config_loss[index] = list()
        d_config = dict()
        d_config['loss'] = item['result']['loss']
        d_config['params_config'] = item['misc']['vals']
        config_loss[index].append(d_config)
    return config_loss


def get_best_parameters(mdl_dict):
    """
    Get param values (defaults as well)
    """
    lr = float(mdl_dict.get("learning_rate", "0.001"))
    embedding_size = int(mdl_dict.get("embedding_size", "512"))
    dropout = float(mdl_dict.get("dropout", "0.2"))
    spatial_dropout = float(mdl_dict.get("spatial_dropout", "0.2"))
    filter_size = float(mdl_dict.get("filter_size", "512"))
    kernel_size = float(mdl_dict.get("kernel_size", "5"))
    deep_size = int(mdl_dict.get("deep_size", "512"))
    batch_size = int(mdl_dict.get("batch_size", "512"))
    deep_activation = mdl_dict.get("deep_activation", "elu")
    output_activation = mdl_dict.get("output_activation", "sigmoid")
    loss_type = mdl_dict.get("loss_type", "binary_crossentropy")
    
    '''lr = float(mdl_dict.get("learning_rate", "0.0012307672546688516"))
    embedding_size = int(mdl_dict.get("embedding_size", "472"))
    dropout = float(mdl_dict.get("dropout", "0.004887220172808515"))
    spatial_dropout = float(mdl_dict.get("spatial_dropout", "0.3325902753016326"))
    filter_size = float(mdl_dict.get("filter_size", "230"))
    kernel_size = float(mdl_dict.get("kernel_size", "8"))
    deep_size = int(mdl_dict.get("deep_size", "78"))
    batch_size = int(mdl_dict.get("batch_size", "485"))
    deep_activation = mdl_dict.get("deep_activation", "elu")
    output_activation = mdl_dict.get("output_activation", "sigmoid")
    loss_type = mdl_dict.get("loss_type", "binary_crossentropy")'''

    return {
        "lr": lr,
        "embedding_size": embedding_size,
        "dropout": dropout,
        "spatial_dropout": spatial_dropout,
        "filter_size": filter_size,
        "kernel_size": kernel_size,
        "deep_size": deep_size,
        "batch_size": batch_size,
        "deep_activation": deep_activation,
        "output_activation": output_activation,
        "loss_type": loss_type
    }


def set_cnn_network(mdl_dict, reverse_dictionary):
    """
    Create a convolutional network and set its parameters
    """
    dimensions = len(reverse_dictionary) + 1
    model_params = get_best_parameters(mdl_dict)

    # define the architecture of the convolutional neural network
    model = Sequential()
    model.add(Embedding(dimensions, model_params["embedding_size"]))
    model.add(SpatialDropout1D(model_params["spatial_dropout"]))
    model.add(Conv1D(int(model_params["filter_size"]), int(model_params["kernel_size"]), activation=model_params['deep_activation']))
    model.add(Dropout(float(model_params["dropout"])))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(int(model_params["deep_size"]), activation=model_params['deep_activation']))
    model.add(Dense(dimensions, activation=model_params['output_activation']))   
    model.compile(loss=model_params["loss_type"], optimizer=RMSprop(lr=model_params['lr']))
    return model, model_params


def compute_precision(model, x, y, reverse_data_dictionary, next_compatible_tools, usage_scores, actual_classes_pos, topk):
    """
    Compute absolute and compatible precision
    """
    absolute_precision = 0.0
    compatible_precision = 0.0
    test_sample = np.reshape(x, (1, len(x)))
    test_sample_pos = np.where(x > 0)[0]
    test_sample_tool_pos = x[test_sample_pos[0]:]

    # predict next tools for a test path
    prediction = model.predict(test_sample, verbose=0)
    nw_dimension = prediction.shape[1]

    # remove the 0th position as there is no tool at this index
    prediction = np.reshape(prediction, (nw_dimension,))

    prediction_pos = np.argsort(prediction, axis=-1)
    topk_prediction_pos = prediction_pos[-topk:]

    # remove the wrong tool position from the predicted list of tool positions
    topk_prediction_pos = [x for x in topk_prediction_pos if x > 0]

    # read tool names using reverse dictionary
    sequence_tool_names = [reverse_data_dictionary[int(tool_pos)] for tool_pos in test_sample_tool_pos]
    actual_next_tool_names = [reverse_data_dictionary[int(tool_pos)] for tool_pos in actual_classes_pos]
    top_predicted_next_tool_names = [reverse_data_dictionary[int(tool_pos)] for tool_pos in topk_prediction_pos]

    # compute the class weights of predicted tools
    mean_usg_score = 0
    usg_wt_scores = list()
    for t_id in topk_prediction_pos:
        t_name = reverse_data_dictionary[int(t_id)]
        if t_id in usage_scores and t_name in actual_next_tool_names:
            usg_wt_scores.append(usage_scores[t_id])
    if len(usg_wt_scores) > 0:
            mean_usg_score = np.mean(usg_wt_scores)
    false_positives = [tool_name for tool_name in top_predicted_next_tool_names if tool_name not in actual_next_tool_names]
    absolute_precision = 1 - (len(false_positives) / float(topk))
    return mean_usg_score, absolute_precision


def verify_model(model, x, y, reverse_data_dictionary, next_compatible_tools, usage_scores, topk_list=[1, 2, 3]):
    """
    Verify the model on test data
    """
    print("Evaluating performance on test data...")
    print("Test data size: %d" % len(y))
    size = y.shape[0]
    precision = np.zeros([len(y), len(topk_list)])
    usage_weights = np.zeros([len(y), len(topk_list)])
    # loop over all the test samples and find prediction precision
    for i in range(size):
        actual_classes_pos = np.where(y[i] > 0)[0]
        for index, abs_topk in enumerate(topk_list):
            abs_mean_usg_score, absolute_precision = compute_precision(model, x[i, :], y, reverse_data_dictionary, next_compatible_tools, usage_scores, actual_classes_pos, abs_topk)
            precision[i][index] = absolute_precision
            usage_weights[i][index] = abs_mean_usg_score
    mean_precision = np.mean(precision, axis=0)
    mean_usage = np.mean(usage_weights, axis=0)
    return mean_precision, mean_usage


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
