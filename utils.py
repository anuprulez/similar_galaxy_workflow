import os
import numpy as np
import json
import h5py

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, Callback
from keras.layers.core import SpatialDropout1D
from keras.optimizers import RMSprop


def save_network( model ):
    """
    Save the network as json file
    """
    with open( NETWORK_C0NFIG_JSON_PATH, "w" ) as json_file:
        json_file.write( model )

def read_file( file_path ):
    """
    Read a file
    """
    with open( file_path, "r" ) as json_file:
        file_content = json.loads( json_file.read() )
    return file_content

def get_h5_data( file_name ):
    """
    Read h5 file to get train and test data
    """
    hf = h5py.File( file_name, 'r' )
    return hf.get( "data" ), hf.get( "data_labels" )

def load_saved_model( network_config_path, weights_path ):
    """
    Load the saved trained model using the saved network and its weights
    """
    with open( network_config_path, 'r' ) as network_config_file:
        loaded_model = network_config_file.read()
        # load the network
        loaded_model = model_from_json(loaded_model)
        # load the saved weights into the model
        loaded_model.load_weights( weights_path )
    return loaded_model

def save_network( model, file_path ):
    """
    Save the network as json file
    """
    with open( file_path, "w" ) as json_file:
        json_file.write( model )

def remove_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
        
def set_recurrent_network(mdl_dict, reverse_dictionary):
    """
    Create a RNN network and set its parameters
    """
    dimensions = len( reverse_dictionary ) + 1
    lr = float(mdl_dict.get("learning_rate", "0.001"))
    embedding_vector_size = int(mdl_dict.get("embedding_vector_size", "128"))
    dropout = float(mdl_dict.get("dropout", ""))
    units = int(mdl_dict.get("memory_units", "128"))
    batch_size = int(mdl_dict.get("batch_size", "128"))
    loss = mdl_dict.get("loss_type", "binary_crossentropy")
    activation_recurrent = mdl_dict.get("activation_recurrent", "elu")
    activation_output = mdl_dict.get("activation_output", "sigmoid")
        
    # define the architecture of the recurrent neural network
    model = Sequential()
    model.add(Embedding(dimensions, embedding_vector_size, mask_zero=True))
    model.add(SpatialDropout1D(dropout))
    model.add(GRU(units, dropout=dropout, recurrent_dropout=dropout, return_sequences=True, activation=activation_recurrent))
    model.add(Dropout(dropout))
    model.add(GRU(units, dropout=dropout, recurrent_dropout=dropout, return_sequences=False, activation=activation_recurrent))
    model.add(Dropout(dropout))
    model.add(Dense(dimensions, activation=activation_output))
    optimizer = RMSprop(lr=lr)
    model.compile(loss=loss, optimizer=optimizer)
    return model
       
def verify_model( model, x, y, reverse_data_dictionary ):
    """
    Verify the model on test data
    """
    print("Evaluating performance on test data...")
    print("Test data size: %d" % len(y))
    size = y.shape[ 0 ]
    topk_abs_pred = np.zeros( [ size ] )
    topk_compatible_pred = np.zeros( [ size ] )
    ave_abs_precision = list()
    # loop over all the test samples and find prediction precision
    for i in range( size ):
        actual_classes_pos = np.where( y[ i ] > 0 )[ 0 ]
        topk = len( actual_classes_pos )
        test_sample = np.reshape( x[ i ], ( 1, x.shape[ 1 ] ) )
        test_sample_pos = np.where( x[ i ] > 0 )[ 0 ]
        test_sample_tool_pos = x[ i ][ test_sample_pos[ 0 ]: ]

        # predict next tools for a test path
        prediction = model.predict( test_sample, verbose=0 )
        nw_dimension = prediction.shape[1]
        
        # remove the 0th position as there is no tool at this index
        prediction = np.reshape(prediction, (nw_dimension,))

        prediction_pos = np.argsort( prediction, axis=-1 )
        topk_prediction_pos = prediction_pos[ -topk: ]

        # read tool names using reverse dictionary
        actual_next_tool_names = [ reverse_data_dictionary[ str( int( tool_pos ) ) ] for tool_pos in actual_classes_pos ]
        top_predicted_next_tool_names = [ reverse_data_dictionary[ str( int( tool_pos ) ) ] for tool_pos in topk_prediction_pos ]

        # find false positives
        false_positives = [ tool_name for tool_name in top_predicted_next_tool_names if tool_name not in actual_next_tool_names ]
        absolute_precision = 1 - ( len( false_positives ) / float( len( actual_classes_pos ) ) )
        ave_abs_precision.append(absolute_precision)
    mean_precision = np.mean(ave_abs_precision)
    print("Absolute precision on test data using current model is: %0.6f" % mean_precision)
    return mean_precision
