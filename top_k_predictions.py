"""
Evaluate the generated models at each epoch using test data
"""
import sys
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt
import csv

# machine learning library
from keras.models import model_from_json


class EvaluateTopResults:

    @classmethod
    def __init__( self ):
        """ Init method. """
        self.current_working_dir = os.getcwd()
        self.network_config_json_path = self.current_working_dir + "/data/model.json"
        self.weights_path = self.current_working_dir + "/data/weights/weights-epoch-50.hdf5"
        self.test_labels_path = self.current_working_dir + "/data/test_data_labels_dict.txt"
        self.train_labels_path = self.current_working_dir + "/data/train_data_labels_dict.txt"
        self.train_class_acc = self.current_working_dir + "/data/train_class_acc.txt"
        self.test_class_acc = self.current_working_dir + "/data/test_class_acc.txt"
        self.data_dictionary_path = self.current_working_dir + "/data/data_dictionary.txt"
        self.data_dictionary_rev_path = self.current_working_dir + "/data/data_rev_dict.txt"
        self.test_class_topk_accuracy = self.current_working_dir + "/data/test_class_topk_accuracy.txt"
        self.train_class_topk_accuracy = self.current_working_dir + "/data/train_class_topk_accuracy.txt"
        self.compatible_tools_filetypes = self.current_working_dir + "/data/compatible_tools.json"
        self.max_seq_length = 40

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
    def get_per_class_topk_acc( self, data, model, dimensions, reverse_data_dictionary, compatible_filetypes, train_labels ):
        """
        Compute average per class topk accuarcy 
        """
        # iterate over all the samples in the data
        data = list( data.items() )
        class_topk_accuracy = list()
        test_data_performance = list()
        min_seq_length = 0
        top1 = 1
        for i in range( len( data ) ):
            topk_prediction = 0.0
            num_class_topk = dict()
            test_seq_performance = dict()
            sequence = list()
            train_seq_padded = np.zeros( [ self.max_seq_length ] )
            train_seq = data[ i ][ 0 ]
            train_seq = train_seq.split( "," )
            for idx, pos in enumerate( train_seq ):
                start_pos = self.max_seq_length - len( train_seq )
                train_seq_padded[ start_pos + idx ] = int( pos )
                sequence.append( pos )
            train_seq_padded = np.reshape( train_seq_padded, ( 1, self.max_seq_length ) )
            prediction = model.predict( train_seq_padded, verbose=0 )
            prediction = np.reshape( prediction, ( dimensions, ) )
            #prediction = prediction[ 1: ]
            prediction_pos = np.argsort( prediction, axis=-1 )
            label_pos = data[ i ][ 1 ].split( "," )
            len_label_pos = top1 #len( label_pos )
            top_prediction_pos = prediction_pos[ -len_label_pos: ]
            top_prediction_pos = [ item for item in reversed( top_prediction_pos ) ]
            actual_tools = [ reverse_data_dictionary[ ps ] for ps in label_pos ]
            predicted_tools = [ reverse_data_dictionary[ str( ps ) ] for ps in top_prediction_pos ]
            false_positives = [ tool for tool in predicted_tools if tool not in actual_tools ]
            sequence_tools = [ reverse_data_dictionary[ ps ] for ps in sequence ]
            seq_string = ",".join( sequence )
            diff_train_labels = ""
            is_prediction_in_train = False
            if seq_string in train_labels:
                diff_train_labels = train_labels[ seq_string ]
                diff_train_labels = [ reverse_data_dictionary[ ps ] for ps in diff_train_labels.split( "," ) ]
                if len( false_positives ) > 0 and false_positives[ 0 ] in diff_train_labels:
                    is_prediction_in_train = True
            for pos in top_prediction_pos:
                if str( pos ) in label_pos:
                    topk_prediction += 1.0
            topk_pred = topk_prediction / float( len( top_prediction_pos ) )
            if len( sequence ) > min_seq_length:
                compatible_tool_types = list()
                test_seq_performance[ "input_sequence" ] = ",".join( sequence_tools )
                test_seq_performance[ "actual_tools" ] = ",".join( actual_tools )
                test_seq_performance[ "predicted_tools" ] = ",".join( predicted_tools )
                test_seq_performance[ "false_positives" ] = ",".join( false_positives )
                test_seq_performance[ "precision" ] = topk_pred
                test_seq_performance[ "labels_in_training" ] = ",".join( diff_train_labels ) if len( diff_train_labels ) > 0 else ""
                test_seq_performance[ "is_prediction_in_train" ] = is_prediction_in_train
                adjusted_precision = topk_pred
                # get the last tool in the input sequence
                seq_last_tool = sequence_tools[ -1 ]
                if seq_last_tool in compatible_filetypes:
                    next_tools = compatible_filetypes[ seq_last_tool ]
                    next_tools = next_tools.split( "," )
                    if len( next_tools ) > 0:
                        for false_pos in false_positives:
                            if false_pos in next_tools:
                                adjusted_precision += 1 / float( len( top_prediction_pos ) )
                                compatible_tool_types.append( false_pos )
                test_seq_performance[ "precision_adjusted_compatibility" ] = adjusted_precision
                test_seq_performance[ "compatible_tool_types" ] = ",".join( compatible_tool_types )
                test_data_performance.append( test_seq_performance )
            num_class_topk[ str( len_label_pos ) ] = topk_pred
            class_topk_accuracy.append( num_class_topk )
        return class_topk_accuracy, test_data_performance
        
    @classmethod
    def save_as_csv( self, list_of_dict, file_name ):
        """
        Save the list of dictionaries as a tabular file
        """
        #keys = list_of_dict[ 0 ].keys()
        # supply actual 
        #keys = [ 'input_sequence', 'actual_tools', 'predicted_tools', 'false_positives', 'compatible_tool_types', 'precision' ]
        #fieldnames = [ "Input tools sequence", "Actual next tools", "Predicted next tools", "False positives", "Compatible tools", "Precision" ]
        keys = [ 'input_sequence', 'actual_tools', 'predicted_tools', 'false_positives', 'compatible_tool_types', 'labels_in_training', "is_prediction_in_train", 'precision', "precision_adjusted_compatibility" ]
        #keys = [ 'input_sequence', 'actual_tools', 'predicted_tools', 'false_positives', 'compatible_tool_types', "precision_adjusted_compatibility" ]
        with open( file_name, 'wb' ) as output_file:
            dict_writer = csv.DictWriter( output_file, keys )
            dict_writer.writeheader()
            dict_writer.writerows( list_of_dict )

    @classmethod
    def get_top_prediction_accuracy( self ):
        """
        Compute top n predictions with a trained model
        """
        loaded_model = self.load_saved_model( self.network_config_json_path, self.weights_path )
        with open( self.test_labels_path, 'r' ) as test_data_labels:
            test_labels = json.loads( test_data_labels.read() )
        with open( self.train_labels_path, 'r' ) as train_data_labels:
            train_labels = json.loads( train_data_labels.read() )
        with open( self.data_dictionary_path, 'r' ) as data_dict:
            data_dict = json.loads( data_dict.read() )
        with open( self.data_dictionary_rev_path, 'r' ) as rev_data_dict:
            reverse_data_dictionary = json.loads( rev_data_dict.read() )
        with open( self.compatible_tools_filetypes, 'r' ) as compatible_file:
            compatible_filetypes = json.loads( compatible_file.read() )
        dimensions = len( data_dict ) + 1
        print ( "Get topn predictions for %d test samples" % len( test_labels ) )
        test_class_topk_accuracy, test_perf = self.get_per_class_topk_acc( test_labels, loaded_model, dimensions, reverse_data_dictionary, compatible_filetypes, train_labels )
        '''with open( self.test_class_topk_accuracy, 'w' ) as test_topk_file:
            test_topk_file.write( json.dumps( test_class_topk_accuracy ) )
        print ( "Get topn predictions for %d train samples" % len( train_labels ) )
        train_class_topk_accuracy, train_perf = self.get_per_class_topk_acc( train_labels, loaded_model, dimensions, reverse_data_dictionary, filetypes )
        train_perf.extend( test_perf )
        with open( self.train_class_topk_accuracy, 'w' ) as train_topk_file:
            train_topk_file.write( json.dumps( train_class_topk_accuracy ) )'''
        self.save_as_csv( test_perf, "data/test_data_performance.csv" )


if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python evaluate_top_results.py" )
        exit( 1 )
    start_time = time.time()
    evaluate_perf = EvaluateTopResults()
    evaluate_perf.get_top_prediction_accuracy()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ) )
