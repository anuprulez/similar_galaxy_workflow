"""
Evaluate the generated models at each epoch using test data
"""
import sys
import numpy as np
import time
import os
import h5py as h5

# machine learning library
from keras.models import model_from_json


class EvaluateTopResults:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.network_config_json_path = self.current_working_dir + "/data/model.json"
        self.base_epochs_weights_path = self.current_working_dir + "/data/weights/weights-epoch-"
        self.test_data_path = self.current_working_dir + "/data/test_data.hdf5"
        self.test_labels_path = self.current_working_dir + "/data/test_labels.hdf5"
        self.top_pred_path = self.current_working_dir + "/data/top_pred.txt"

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
    def evaluate_topn_epochs( self ):
        """
        Get topn accuracy over training epochs
        """
        n_epochs = 700
        num_predictions = 5
        test_data = h5.File( self.test_data_path, 'r' )
        test_data = test_data[ "testdata" ]
        test_labels = h5.File( self.test_labels_path, 'r' )
        test_labels = test_labels[ "testlabels" ]
        topn_accuracy = list()
        dimensions = len( test_labels[ 0 ] )
        for i in range( n_epochs ):
            ite = '0' + str( i + 1 ) if i < 9 else str( i + 1  )
            file_path = self.base_epochs_weights_path + ite + '.hdf5'
            print ( file_path )
            loaded_model = self.load_saved_model( self.network_config_json_path, file_path )
            accuracy = self.get_top_prediction_accuracy( num_predictions, dimensions, loaded_model, test_data, test_labels )
            topn_accuracy.append( accuracy )
            print ( accuracy )
        np.savetxt( self.top_pred_path, np.array( topn_accuracy ), delimiter="," )

    @classmethod
    def get_top_prediction_accuracy( self, topn, dimensions, trained_model, test_data, test_labels ):
        """
        Compute top n predictions with a trained model
        """
        print ( "Get top %d predictions for each test input..." % topn )
        num_predict = len( test_data )
        prediction_accuracy = 0
        for i in range( num_predict ):
            input_seq = test_data[ i ]
            label = test_labels[ i ]
            label_pos = np.where( label > 0 )[ 0 ]
            label_pos = label_pos[ 0 ]
            input_seq_reshaped = np.reshape( input_seq, ( 1, 1, len( input_seq ) ) )
            # predict the next tool using the trained model
            prediction = trained_model.predict( input_seq_reshaped, verbose=0 )
            prediction = np.reshape( prediction, ( dimensions ) )
            # take prediction in reverse order, best ones first
            prediction_pos = np.argsort( prediction, axis=0 )
            # get top n predictions
            top_prediction_pos = prediction_pos[ -topn: ]
            # get tool names for the predicted positions
            top_predictions = [ pred_pos for pred_pos in top_prediction_pos if label_pos == pred_pos ]
            if len( top_predictions ) > 0:
                prediction_accuracy += 1
        return float( prediction_accuracy ) / num_predict


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python evaluate_top_results.py" )
        exit( 1 )
    start_time = time.time()
    evaluate_perf = EvaluateTopResults()
    evaluate_perf.evaluate_topn_epochs()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ) )
