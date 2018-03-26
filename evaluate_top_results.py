"""
Evaluate the generated models at each epoch using test data
"""
import sys
import numpy as np
import time
import os
import h5py as h5
import json

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
        self.abs_top_pred_path = self.current_working_dir + "/data/abs_top_pred.txt"
        self.train_test_labels = self.current_working_dir + "/data/multi_labels.txt"

    @classmethod
    def load_saved_model( self, network_config_path, weights_path ):
        """
        Load the saved trained model using the saved network and its weights
        """
        with open( network_config_path, 'r' ) as network_config_file:
            loaded_model = network_config_file.read()
        # load the network
        loaded_model = model_from_json( loaded_model )
        # load the saved weights into the model
        loaded_model.load_weights( weights_path )
        return loaded_model

    @classmethod
    def evaluate_topn_epochs( self ):
        """
        Get topn accuracy over training epochs
        """
        n_epochs = 50
        num_predictions = 5
        test_data = h5.File( self.test_data_path, 'r' )
        test_data = test_data[ "testdata" ]
        test_labels = h5.File( self.test_labels_path, 'r' )
        test_labels = test_labels[ "testlabels" ]
        topn_accuracy = list()
        abs_topn_accuracy = list()
        dimensions = len( test_labels[ 0 ] )
        for i in range( n_epochs ):
            start_time = time.time()
            ite = '0' + str( i + 1 ) if i < 9 else str( i + 1  )
            file_path = self.base_epochs_weights_path + ite + '.hdf5'
            loaded_model = self.load_saved_model( self.network_config_json_path, file_path )
            accuracy, abs_mutual_prediction_accuracy = self.get_top_prediction_accuracy( num_predictions, dimensions, loaded_model, test_data, test_labels )
            topn_accuracy.append( np.mean( accuracy ) )
            abs_topn_accuracy.append( np.mean( abs_mutual_prediction_accuracy ) )
            print ( np.mean( accuracy ) )
            print ( np.mean( abs_mutual_prediction_accuracy ) )
            end_time = time.time()
            print( "Prediction finished in %d seconds for epoch %d" % ( int( end_time - start_time ), i + 1 ) )
            print( "========================" )
        np.savetxt( self.top_pred_path, np.array( topn_accuracy ), delimiter="," )
        np.savetxt( self.abs_top_pred_path, np.array( abs_topn_accuracy ), delimiter="," )

    @classmethod
    def get_top_prediction_accuracy( self, topn, dimensions, trained_model, test_data, test_labels ):
        """
        Compute top n predictions with a trained model
        """
        print ( "Get top %d predictions for each test input..." % topn )
        num_predict = len( test_data )
        mutual_prediction_accuracy = np.zeros( [ num_predict ] )
        abs_mutual_prediction_accuracy = np.zeros( [ num_predict ] )
        with open( self.train_test_labels, 'r' ) as train_data_labels:
            data_labels = json.loads( train_data_labels.read() )
        # iterate over all the samples in the test data
        for i in range( num_predict ):
            mutual_prediction = 0.0
            abs_mutual_prediction = 0.0
            actual_labels = list()
            input_seq = test_data[ i ]
            label = test_labels[ i ]
            label_pos = np.where( label > 0 )[ 0 ]
            label_pos = label_pos[ 0 ]
            input_seq_reshaped = np.reshape( input_seq, ( 1, len( input_seq ) ) )
            input_seq_pos = np.where( input_seq > 0 )[ 0 ]
            input_seq_list = [ str( int( input_seq[ item ] + 1 ) ) for item in input_seq_pos ]
            input_seq_list = ",".join( input_seq_list )
            # predict the next tool using the trained model
            prediction = trained_model.predict( input_seq_reshaped, verbose=0 )
            prediction = np.reshape( prediction, ( dimensions, ) )
            # take prediction in reverse order, best ones first
            prediction_pos = np.argsort( prediction, axis=0 )
            # get the actual labels for the input sequence
            if input_seq_list in data_labels:
                actual_labels = data_labels[ input_seq_list ]
            if actual_labels:
                actual_labels = actual_labels.split( "," )
            num_actual_labels = len( actual_labels )
            if num_actual_labels > 0:
                # get top n predictions
                top_prediction_pos = prediction_pos[ -topn: ]
                top_prediction_pos = [ ( item + 1 ) for item in reversed( top_prediction_pos ) ]
                # find how many of all the true labels (k) present in the top-k predicted labels
                abs_top_prediction_pos = prediction_pos[ -num_actual_labels: ]
                abs_top_prediction_pos = [ ( item + 1 ) for item in reversed( abs_top_prediction_pos ) ]
                # find how many actual labels are present in the predicted ones
                for item in actual_labels:
                    if int( item ) in top_prediction_pos:
                        mutual_prediction += 1.0
                    if int( item ) in abs_top_prediction_pos:
                        abs_mutual_prediction += 1.0
                pred = mutual_prediction / float( num_actual_labels )
                abs_pred = abs_mutual_prediction / float( num_actual_labels )      
                abs_mutual_prediction_accuracy[ i ] = abs_pred
                mutual_prediction_accuracy[ i ] = pred
        return mutual_prediction_accuracy, abs_mutual_prediction_accuracy


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python evaluate_top_results.py" )
        exit( 1 )
    start_time = time.time()
    evaluate_perf = EvaluateTopResults()
    evaluate_perf.evaluate_topn_epochs()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ) )
