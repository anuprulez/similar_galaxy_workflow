"""
Evaluate the generated models at each epoch using test data
"""
import sys
import numpy as np
import time
import os

# machine learning library
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json


class EvaluateTopResults:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.network_config_json_path = self.current_working_dir + "/data/model.json"
        self.weights_path = self.current_working_dir + "/data/weights/trained_model.h5"
        self.base_epochs_weights_path = self.current_working_dir + "/data/weights/weights-epoch-"

    @classmethod
    def load_saved_model( self, network_config_path, weights_path ):
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

    @classmethod
    def evaluate_topn_epochs( self, n_epochs, num_predictions, dimensions, reverse_dictionary, test_data, test_labels ):
        """
        Get topn accuracy over training epochs
        """
        topn_accuracy = list()
        for i in range( n_epochs ):
            ite = '0' + str( i + 1 ) if i < 9 else str( i + 1  )
            file_path = self.base_epochs_weights_path + ite + '.hdf5'
            print file_path
            loaded_model = self.load_saved_model( self.network_config_json_path, file_path )
            accuracy = self.get_top_prediction_accuracy( num_predictions, dimensions, loaded_model, reverse_dictionary, test_data, test_labels )
            topn_accuracy.append( accuracy )
            print accuracy
        print topn_accuracy

    @classmethod
    def get_top_prediction_accuracy( self, topn, dimensions, trained_model, reverse_dictionary, test_data, test_labels ):
        """
        Compute top n predictions with a trained model
        """
        print "Get top %d predictions for each test input..." % topn
        num_predict = len( test_data )
        prediction_accuracy = 0
        for i in range( num_predict ):
            input_seq = test_data[ i ][ 0 ]
            label = test_labels[ i ][ 0 ]
            #print label.shape
            label_pos = np.where( label > 0 )[ 0 ]
            #print label_pos
            label_text = reverse_dictionary[ label_pos[ 0 ] ]
            #print label_text
            input_seq_reshaped = np.reshape( input_seq, ( 1, 1, dimensions ) )
            # predict the next tool using the trained model
            prediction = trained_model.predict( input_seq_reshaped, verbose=0 )
            prediction = np.reshape( prediction, ( dimensions ) )
            # take prediction in reverse order, best ones first
            prediction_pos = np.argsort( prediction, axis=0 )
            # get top n predictions
            top_prediction_pos = prediction_pos[ -topn: ]
            # get tool names for the predicted positions
            top_predictions = [ reverse_dictionary[ pred_pos ] for pred_pos in top_prediction_pos ]
            top_predicted_tools_text = " ".join( top_predictions )
            if label_text in top_predictions:
                prediction_accuracy += 1
        return float( prediction_accuracy ) / num_predict
