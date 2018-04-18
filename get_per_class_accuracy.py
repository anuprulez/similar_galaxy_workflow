"""
Evaluate the generated models at each epoch using test data
"""
import sys
import numpy as np
import time
import os
import json
import matplotlib.pyplot as plt

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
        self.test_class_topk_accuracy = self.current_working_dir + "/data/test_class_topk_accuracy.txt"
        self.train_class_topk_accuracy = self.current_working_dir + "/data/train_class_topk_accuracy.txt"
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
    def get_per_class_topk_acc( self, data, model, dimensions ):
        """
        Compute average per class topk accuarcy 
        """
        # iterate over all the samples in the data
        data = list( data.items() )
        class_topk_accuracy = list()
        for i in range( len( data ) ):
            topk_prediction = 0.0
            num_class_topk = dict()
            
            train_seq_padded = np.zeros( [ self.max_seq_length ] )
            train_seq = data[ i ][ 0 ]
            train_seq = train_seq.split( "," )
            for idx, pos in enumerate( train_seq ):
                start_pos = self.max_seq_length - len( train_seq )
                train_seq_padded[ start_pos + idx ] = int( pos ) - 1
            train_seq_padded = np.reshape( train_seq_padded, ( 1, self.max_seq_length ) )
            prediction = model.predict( train_seq_padded, verbose=0 )
            prediction = np.reshape( prediction, ( dimensions, ) )
            prediction_pos = np.argsort( prediction, axis=0 )
            
            label = data[ i ][ 1 ]
            label_pos = label.split( "," )
            len_label_pos = len( label_pos )
            top_prediction_pos = prediction_pos[ -len_label_pos: ]
            top_prediction_pos = [ ( item + 1 ) for item in reversed( top_prediction_pos ) ]
            for pos in top_prediction_pos:
                if str( pos ) in label_pos:
                    topk_prediction += 1.0
            topk_pred = topk_prediction / float( len( top_prediction_pos ) )
            num_class_topk[ str( len_label_pos ) ] = topk_pred
            class_topk_accuracy.append( num_class_topk )
        return class_topk_accuracy

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
        dimensions = len( data_dict )
        print ( "Get topn predictions for each test sample..." )
        test_class_topk_accuracy = self.get_per_class_topk_acc( test_labels, loaded_model, dimensions )
        with open( self.test_class_topk_accuracy, 'w' ) as test_topk_file:
            test_topk_file.write( json.dumps( test_class_topk_accuracy ) )
        train_class_topk_accuracy = self.get_per_class_topk_acc( train_labels, loaded_model, dimensions )
        with open( self.train_class_topk_accuracy, 'w' ) as train_topk_file:
            train_topk_file.write( json.dumps( train_class_topk_accuracy ) )
        
    @classmethod
    def plot_per_class_topk_accuracy( self ):
        """
        Plot the accuracy 
        """
        with open( self.test_class_topk_accuracy, 'r' ) as test_topk_file:
            test_acc = json.loads( test_topk_file.read() )
        
        with open( self.train_class_topk_accuracy, 'r' ) as train_topk_file:
            train_acc = json.loads( train_topk_file.read() )

        classes = list()
        accuracies = list()
        classes_train = list()
        classes_acc = dict()
        accuracies_train = list()
        classes_count = dict()
        classes_acc_train = dict()
        classes_count_train = dict()
        count_classes_test = list()
        count_classes_train = list()
        for index, item in enumerate( test_acc ):
            for key in item:
                if key in classes_acc:
                    classes_acc[ key ] += float( item[ key ] )
                    classes_count[ key ] += 1
                else:
                    classes_acc[ key ] = float( item[ key ] )
                    classes_count[ key ] = 1
                    
        for item in classes_acc:
            classes_acc[ item ] = classes_acc[ item ] / float( classes_count[ item ] )
        for item in classes_acc:
            acc = classes_acc[ item ]
            accuracies.append( acc )
            classes.append( int( item ) )
            count_classes_test.append( classes_count[ item ] )
          
        # training data
        for item in train_acc:
            for key in item:
                if key in classes_acc_train:
                    classes_acc_train[ key ] += float( item[ key ] )
                    classes_count_train[ key ] += 1
                else:
                    classes_acc_train[ key ] = float( item[ key ] )
                    classes_count_train[ key ] = 1
                    
        for item in classes_acc_train:
            classes_acc_train[ item ] = classes_acc_train[ item ] / float( classes_count_train[ item ] )
        for item in classes_acc_train:
            accuracies_train.append( classes_acc_train[ item ] )
            classes_train.append( int( item ) )
            count_classes_train.append( classes_count_train[ item ] )
        font = { 'family' : 'sans serif', 'size': 22 }
        plt.rc('font', **font)
        plt.plot( classes, accuracies, 'ro' )
        plt.plot( classes_train, accuracies_train, 'bo' )
        plt.xlabel( 'Number of classes (next tools)' )
        plt.ylabel( 'Topk accuracy ( 0.7 = 70% )' )
        plt.title( 'Number of classes vs avg. topk accuracy for all samples' )
        plt.legend([ "Test samples", "Train samples" ])
        plt.grid( True )
        plt.show()
        
        plt.bar( classes, count_classes_test, color='red' )
        plt.xlabel( 'Number of classes (next tools)' )
        plt.ylabel( 'Number of samples' )
        plt.title( 'Number of samples per class (next tool) for test samples' )
        plt.grid( True )
        plt.show()
        
        plt.bar( classes_train, count_classes_train, color='blue' )
        plt.xlabel( 'Number of classes (next tools)' )
        plt.ylabel( 'Number of samples' )
        plt.title( 'Number of samples per class (next tool) for train samples' )
        plt.grid( True )
        plt.show()

if __name__ == "__main__":

    if len(sys.argv) != 1:
        print( "Usage: python evaluate_top_results.py" )
        exit( 1 )
    start_time = time.time()
    evaluate_perf = EvaluateTopResults()
    evaluate_perf.get_top_prediction_accuracy()
    evaluate_perf.plot_per_class_topk_accuracy()
    end_time = time.time()
    print ("Program finished in %s seconds" % str( end_time - start_time ) )
